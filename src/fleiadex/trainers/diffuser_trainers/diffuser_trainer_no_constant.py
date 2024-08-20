import shutil

import jax.sharding
from absl import logging
from functools import partial
import os
from flax import linen as nn
from flax.core import FrozenDict
from jax import random
import jax.numpy as jnp
import einops
import optax
from clu import platform
from jax.lib import xla_bridge
import tensorflow as tf
import hydra.utils as hydra_utils
from einops import rearrange

import orbax.checkpoint as ocp
import etils.epath as path

import hydra
from omegaconf import OmegaConf

from fleiadex.nn_models import VAE
from fleiadex.utils import mse, TrainStateWithDropout, save_many_images, save_image
from fleiadex.diffuser import DDPMCore
from fleiadex.config import LDMConfig


class Trainer:
    def __init__(self, config: LDMConfig):

        tf.config.experimental.set_visible_devices([], "GPU")
        logging.info(f"JAX backend: {xla_bridge.get_backend().platform}")

        logging.info(f"JAX process: {jax.process_index() + 1} / {jax.process_count()}")
        logging.info(f"JAX local devices: {jax.local_devices()}")

        platform.work_unit().set_task_status(
            f"process_index: {jax.process_index()}, "
            f"process_count: {jax.process_count()}"
        )

        # convert to FrozenDict, the standard config container in jax
        self.config: FrozenDict = FrozenDict(OmegaConf.to_container(config))

        jax.config.update('jax_debug_nans', self.config['global_config']['debug_nans'])

        # initialise the random number generator keys
        (
            self.const_rng,
            self.eval_rng,
            self.dropout_rng,
            self.train_key,
        ) = self._init_rng()
        # train_key is only used to split keys
        self.train_key, self.train_rng = random.split(self.train_key)

        # the diffusion core
        self.diffuser = DDPMCore(
            config=self.config,
            diffusion_time_steps=self.config["hyperparams"]["diffusion_time_steps"],
        )

        logging.info("initializing dataset.")
        self.data_loader = hydra_utils.instantiate(config.data_spec, _recursive_=False)

        (
            self.train_ds,
            self.test_ds,
        ) = self.data_loader.write_data_summary().get_train_test_dataset()

        # initialise the save directory
        self.save_dir = self._init_savedir()

        # optimiser with exponential moving average
        self.optimiser = self._get_optimiser(self.config)
        self.optimiser = optax.chain(
            optax.clip_by_global_norm(self.config['hyperparams']['gradient_clipping']),
            self.optimiser,
            optax.ema(self.config['hyperparams']['ema_decay']),
        )

        # indicator of vae availability
        self.__vae_ready__ = False
        self.vae = None

        # time frame length
        self.temporal_length = (
            self.config["data_spec"]["condition_length"]
            + self.config["data_spec"]["prediction_length"]
        )

        # calculate prediction saving frame length
        self.save_length = (self.config["data_spec"]["condition_length"] +
                            3*self.config["data_spec"]["prediction_length"])
        self.save_length = self.save_length if self.save_length < 10 else 10

        # load vae if not encoded
        if self.config['data_spec']['pre_encoded'] is False:
            if self.config['hyperparams']['load_vae_dir'] is not None:
                self.load_vae_from(self.config['hyperparams']['load_vae_dir'])
                self.__vae_ready__ = True
            else:
                raise ValueError('for unencoded dataset, vae must be specified.')

        self.__manual_load__ = False
        if self.config['hyperparams']['load_ckpt_dir'] is not None:
            self.load_diffuser_ckpt_from(self.config['hyperparams']['load_ckpt_dir'],
                                         self.config['hyperparams']['load_ckpt_step'],
                                         self.config['hyperparams']['load_ckpt_config'])
            self.__manual_load__ = True

    def _init_rng(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        init_key = random.PRNGKey(42)
        const_rng, eval_rng, dropout_rng, train_key = random.split(init_key, 4)
        return const_rng, eval_rng, dropout_rng, train_key

    def _init_savedir(self) -> str:
        save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        save_dir = str(os.path.join(save_dir, "results"))
        os.makedirs(save_dir)
        os.makedirs(save_dir + "/predictions")
        return save_dir

    def _get_optimiser(self, config: FrozenDict) -> optax.GradientTransformation:
        if isinstance(config["hyperparams"]["learning_rate"], float):
            return optax.adam(config["hyperparams"]["learning_rate"])
        else:
            try:
                optimiser = optax.adam(
                    eval(config["hyperparams"]["learning_rate"], {"optax": optax})
                )
            except Exception as e:
                raise ValueError(
                    f"unknown learning rate type: {e} \nplease follow optax syntax."
                )
            return optimiser

    def _update_train_rng(self) -> None:
        self.train_key, self.train_rng = random.split(self.train_key)

    def _update_eval_rng(self) -> None:
        self.eval_rng, _ = random.split(self.eval_rng)

    def _get_next_train_batch(self):
        batch = next(self.train_ds)
        b, t, _, _, _ = batch.shape
        batch = einops.rearrange(batch, "b t h w c -> (b t) h w c")
        batch = self.vae.apply_fn(
            {"params": self.vae.params},
            batch,
            self.eval_rng,
            rngs={"dropout": self.dropout_rng},
            method="encode",
        )
        batch = batch.reshape(b, t, batch.shape[1], batch.shape[2], batch.shape[3])

        return batch

    def _get_next_train_batch_pre_encoded(self):
        return next(self.train_ds)

    def _get_next_test_batch(self):
        batch = next(self.test_ds)
        b, t, _, _, _ = batch.shape
        batch = einops.rearrange(batch, "b t h w c -> (b t) h w c")
        batch = self.vae.apply_fn(
            {"params": self.vae.params},
            batch,
            self.eval_rng,
            rngs={"dropout": self.dropout_rng},
            method="encode",
        )
        batch = batch.reshape(b, t, batch.shape[1], batch.shape[2], batch.shape[3])

        return batch

    def _get_next_test_batch_pre_coded(self):
        return next(self.test_ds)

    @partial(jax.jit, static_argnames=("self",))
    def _train_step(self, state, batch):
        def loss_fn(params):
            pred_noise, true_noise = state.apply_fn(
                {"params": params},
                batch,
                self.train_rng,
                True,
                rngs={"dropout": self.dropout_rng},
            )

            mse_loss = mse(pred_noise, true_noise).mean()

            return mse_loss

        grads = jax.grad(loss_fn)(state.params)
        # updates, state = self.optimiser.update(grads, state, state.params)
        # state.params = optax.apply_updates(state.params, updates)
        state = state.apply_gradients(grads=grads)

        return state

    def load_vae_from(self, ckpt_dir):
        logging.info("loading vae from %s", ckpt_dir)

        if isinstance(ckpt_dir, str):
            ckpt_dir = path.Path(ckpt_dir)
        else:
            raise ValueError("checkpoint directory must be a string.")

        mngr = ocp.CheckpointManager(ckpt_dir, item_names=("vae_state", "config"))

        restored_config = mngr.restore(
            mngr.latest_step(), args=ocp.args.Composite(config=ocp.args.JsonRestore())
        )

        vae_config = FrozenDict(restored_config["config"])
        vae = VAE(**vae_config["nn_spec"])

        init_data = jnp.ones(
            (
                self.config["hyperparams"]["batch_size"],
                self.config["nn_spec"]["sample_input_shape"][0],
                self.config["nn_spec"]["sample_input_shape"][1],
                self.config["nn_spec"]["sample_input_shape"][2],
            ),
            jnp.float32,
        )

        rngs = {"params": random.PRNGKey(0), "dropout": random.PRNGKey(42)}
        params = vae.init(rngs, init_data, random.key(0), False)["params"]

        vae_optimiser = optax.chain(
            optax.clip_by_global_norm(vae_config['hyperparams']['gradient_clipping']),
            self._get_optimiser(vae_config), )

        vae_state = TrainStateWithDropout.create(
            apply_fn=vae.apply,
            params=params,
            tx=vae_optimiser,
            key=self.dropout_rng,
        )


        sharding = jax.sharding.NamedSharding(
            mesh=jax.sharding.Mesh(jax.devices(), axis_names="model"),
            spec=jax.sharding.PartitionSpec(),
        )

        create_sharded_array = lambda x: jax.device_put(x, sharding)
        vae_state = jax.tree_util.tree_map(create_sharded_array, vae_state)
        vae_state = jax.tree_util.tree_map(
            ocp.utils.to_shape_dtype_struct, vae_state
        )

        restored = mngr.restore(
            mngr.latest_step(),
            args=ocp.args.Composite(vae_state=ocp.args.StandardRestore(vae_state)),
        )["vae_state"]

        logging.info("loading succeeded.")

        self.__vae_ready__ = True
        self.vae = restored
        del mngr

    @partial(jax.jit, static_argnums=0)
    def _evaluate(self, params, test_batch):
        def compute_metric(pred, true) -> dict[str, jnp.ndarray]:
            mse_loss = mse(pred, true).mean()
            return {"mse": mse_loss, "loss": mse_loss}

        def evaluate(diffuser):

            pred, true, t = diffuser.apply(
                {"params": params},
                test_batch,
                self.eval_rng,
                False, return_t=True,
                rngs={"dropout": self.dropout_rng},
            )

            expect = test_batch[:, self.config["data_spec"]["condition_length"]:]
            noised_expect = diffuser.apply(
                {"params": params},
                expect, true, t,
                rngs={"dropout": self.dropout_rng},
                method="add_noise_to"
            )

            denoised_expect = diffuser.apply(
                {"params": params},
                noised_expect, pred, t, self.eval_rng,
                rngs={"dropout": self.dropout_rng},
                method="denoise"
            )

            predictions = jnp.concatenate(
                [test_batch[:, :self.config["data_spec"]["condition_length"]],
                 expect, noised_expect, denoised_expect],
                axis=1,
            )

            predictions = self.vae.apply_fn(
                {"params": self.vae.params}, predictions, method="decode"
            )

            predictions = rearrange(predictions, 'b t w h c -> (b t) w h c')

            metrics = compute_metric(pred, true)
            return metrics, predictions

        return nn.apply(evaluate, self.diffuser)(
            {"params": params}, rngs={"dropout": self.dropout_rng}
        )

    @partial(jax.jit, static_argnums=0)
    def _evaluate_pre_encoded(self, params, test_batch):
        def compute_metric(pred, true) -> dict[str, jnp.ndarray]:
            mse_loss = mse(pred, true).mean()
            return {"mse": mse_loss, "loss": mse_loss}

        def evaluate(diffuser):

            pred, true, t = diffuser.apply(
                {"params": params},
                test_batch,
                self.eval_rng,
                False, return_t=True,
                rngs={"dropout": self.dropout_rng},
            )

            expect = test_batch[:, self.config["data_spec"]["condition_length"]:]
            noised_expect = diffuser.apply(
                {"params": params},
                expect, true, t,
                rngs={"dropout": self.dropout_rng},
                method="add_noise_to"
            )

            denoised_expect = diffuser.apply(
                {"params": params},
                noised_expect, pred, t, self.eval_rng,
                rngs={"dropout": self.dropout_rng},
                method="denoise"
            )

            predictions = jnp.concatenate(
                [test_batch[:, :self.config["data_spec"]["condition_length"]],
                 expect, noised_expect, denoised_expect],
                axis=1,
            )

            predictions = rearrange(predictions, 'b t w h c -> (b t) w h c')

            metrics = compute_metric(pred, true)
            return metrics, predictions

        return nn.apply(evaluate, self.diffuser)(
            {"params": params}, rngs={"dropout": self.dropout_rng}
        )

    def _save_output(self, save_dir, prediction, step):
        if self.config["hyperparams"]["save_prediction"]:
            save_image(
                save_dir + "/predictions",
                prediction,
                f"/prediction_{int(step + 1)}.png",
                nrow=self.save_length,
                mode=self.config["global_config"]["save_format"],
            )

    def _clear_result(self):
        directory = self.save_dir + "/predictions"
        shutil.rmtree(directory)
        os.makedirs(directory)

    def load_diffuser_ckpt_from(self, ckpt_dir: str, step: int | None = None,
                                load_config: bool = True):
        logging.info(f"loading 4d conditional diffuser from {ckpt_dir}")

        if isinstance(ckpt_dir, str):
            ckpt_dir = path.Path(ckpt_dir)
        else:
            raise ValueError("ckpt dir must be str.")

        mngr = ocp.CheckpointManager(ckpt_dir, item_names=("diffuser_state", "config"))

        if step is not None:
            restored_config = mngr.restore(
                step, args=ocp.args.Composite(config=ocp.args.JsonRestore())
            )
        else:
            restored_config = mngr.restore(
                mngr.latest_step(),
                args=ocp.args.Composite(config=ocp.args.JsonRestore()),
            )

        if load_config:
            self.config = FrozenDict(restored_config["config"])

        self.diffuser = DDPMCore(
            config=self.config,
            diffusion_time_steps=self.config["hyperparams"]["diffusion_time_steps"],
        )

        init_data = jnp.ones(
            (
                self.config["hyperparams"]["batch_size"],
                self.temporal_length,
                self.config["nn_spec"]["sample_input_shape"][1],
                self.config["nn_spec"]["sample_input_shape"][2],
                self.config["nn_spec"]["sample_input_shape"][3],
            ),
            jnp.float32,
        )

        rngs = {"params": random.PRNGKey(0), "dropout": random.PRNGKey(42)}
        params = self.diffuser.init(rngs, init_data, random.key(0), False)["params"]

        self.diffuser_optimiser = self._get_optimiser(self.config)
        self.diffuser_optimiser = optax.chain(
            optax.clip_by_global_norm(self.config['hyperparams']['gradient_clipping']),
            self.diffuser_optimiser,
            optax.ema(self.config['hyperparams']['ema_decay']),
        )

        diffuser_state = TrainStateWithDropout.create(
            apply_fn=self.diffuser.apply,
            params=params,
            tx=self.diffuser_optimiser,
            key=self.train_key,
        )

        sharding = jax.sharding.NamedSharding(
            mesh=jax.sharding.Mesh(jax.devices(), axis_names="model"),
            spec=jax.sharding.PartitionSpec(),
        )

        create_sharded_array = lambda x: jax.device_put(x, sharding)
        diffuser_state = jax.tree_util.tree_map(create_sharded_array, diffuser_state)
        diffuser_state = jax.tree_util.tree_map(
            ocp.utils.to_shape_dtype_struct, diffuser_state
        )

        if step is not None:
            restored = mngr.restore(
                step,
                args=ocp.args.Composite(diffuser_state=ocp.args.StandardRestore(diffuser_state)),
            )["diffuser_state"]
        else:
            restored = mngr.restore(
                mngr.latest_step(),
                args=ocp.args.Composite(diffuser_state=ocp.args.StandardRestore(diffuser_state)),
            )["diffuser_state"]

        self.diffuser_state = restored

        self.__manual_load__ = True

        logging.info("loading succeeded.")
        del mngr

    def train(self, force_visualisation: bool = False):
        if self.config["data_spec"]["pre_encoded"] is True:
            if self.__vae_ready__ is True:
                logging.warning(
                    "since the data is pre-encoded, the loaded VAE will be ignored."
                )
                del self.vae

            if force_visualisation:
                logging.warning(
                    "visualisation was forced to be enabled. may cause unexpected behaviour."
                )
            else:
                logging.warning("visualisation will be disabled.")

            logging.info("initializing model.")
            init_data = jnp.ones(
                (
                    self.config["hyperparams"]["batch_size"],
                    self.temporal_length,
                    self.config["nn_spec"]["sample_input_shape"][1],
                    self.config["nn_spec"]["sample_input_shape"][2],
                    self.config["nn_spec"]["sample_input_shape"][3],
                ),
                jnp.float32,
            )

            rngs = {
                "params": random.PRNGKey(0),
                "dropout": random.PRNGKey(42),
                "constants": self.const_rng,
            }
            collections = self.diffuser.init(rngs, init_data, self.train_rng, False)
            params = collections["params"]

            state = TrainStateWithDropout.create(
                apply_fn=self.diffuser.apply,
                params=params,
                tx=self.optimiser,
                key=self.dropout_rng,
            )

            if self.__manual_load__:
                state = self.diffuser_state

            sharding = jax.sharding.NamedSharding(
                    mesh=jax.sharding.Mesh(jax.devices(), axis_names="model"),
                    spec=jax.sharding.PartitionSpec(),
                )
            create_sharded_array = lambda x: jax.device_put(x, sharding)
            state = jax.tree_util.tree_map(create_sharded_array, state)

            save_path = ocp.test_utils.erase_and_create_empty(
                os.path.abspath(self.save_dir + "/ckpt")
            )
            save_options = ocp.CheckpointManagerOptions(
                max_to_keep=self.config["global_config"]["save_num_ckpts"],
                save_interval_steps=self.config["hyperparams"]["ckpt_freq"],
                best_fn=lambda metrics: float(metrics),
                best_mode='min'
            )

            if self.config["hyperparams"]["save_ckpt"]:
                diffusor_mngr = ocp.CheckpointManager(
                    save_path,
                    options=save_options,
                    item_names=("diffuser_state", "config"),
                )

            for step in range(self.config["hyperparams"]["step"] + 1):

                batch = self._get_next_train_batch_pre_encoded()
                self._update_train_rng()

                state = self._train_step(state, batch)

                eval_count = 0
                if (step % self.config['hyperparams']['eval_freq'] == 0):
                    current_test = self._get_next_test_batch_pre_coded()
                    self._update_eval_rng()

                    metrics, prediction = self._evaluate_pre_encoded(
                        state.params, current_test
                    )

                    logging.info(
                        "step: {}, loss: {:.4f}, mse: {:.4f}".format(
                            step, metrics["loss"], metrics["mse"]
                        )
                    )

                    eval_count += 1
                    if force_visualisation:
                        self._save_output(self.save_dir, prediction, step)

                        if (eval_count % 1000 == 0) and (step != 0):
                            self._clear_result()

                if self.config["hyperparams"]["save_ckpt"]:
                    diffusor_mngr.save(
                        step,
                        args=ocp.args.Composite(
                            diffuser_state=ocp.args.StandardSave(state),
                            config=ocp.args.JsonSave(self.config.unfreeze()),
                        ),
                        metrics=float(metrics["metrics"]),
                    )

            diffusor_mngr.wait_until_finished()

        else:
            if self.__vae_ready__ is False:
                raise ValueError(
                    "vae is not loaded. please run load_vae_from() method first."
                )

            logging.info("initializing model.")
            init_data = jnp.ones(
                (
                    self.config["hyperparams"]["batch_size"],
                    self.temporal_length,
                    self.config["nn_spec"]["sample_input_shape"][1],
                    self.config["nn_spec"]["sample_input_shape"][2],
                    self.config["nn_spec"]["sample_input_shape"][3],
                ),
                jnp.float32,
            )

            rngs = {
                "params": random.PRNGKey(0),
                "dropout": random.PRNGKey(42),
                "constants": self.const_rng,
            }
            collections = self.diffuser.init(rngs, init_data, self.train_rng, False)
            params = collections["params"]

            state = TrainStateWithDropout.create(
                apply_fn=self.diffuser.apply,
                params=params,
                tx=self.optimiser,
                key=self.dropout_rng,
            )

            if self.__manual_load__:
                state = self.diffuser_state

            sharding = jax.sharding.NamedSharding(
                    mesh=jax.sharding.Mesh(jax.devices(), axis_names="model"),
                    spec=jax.sharding.PartitionSpec(),
                )
            create_sharded_array = lambda x: jax.device_put(x, sharding)
            state = jax.tree_util.tree_map(create_sharded_array, state)

            save_path = ocp.test_utils.erase_and_create_empty(
                os.path.abspath(self.save_dir + "/ckpt")
            )
            save_options = ocp.CheckpointManagerOptions(
                max_to_keep=self.config["global_config"]["save_num_ckpts"],
                save_interval_steps=self.config["hyperparams"]["ckpt_freq"],
                best_fn=lambda metrics: float(metrics),
                best_mode='min'
            )

            if self.config["hyperparams"]["save_ckpt"]:
                diffusor_mngr = ocp.CheckpointManager(
                    save_path,
                    options=save_options,
                    item_names=("diffuser_state", "config"),
                )

            for step in range(self.config["hyperparams"]["step"] + 1):

                batch = self._get_next_train_batch()
                self._update_train_rng()

                state = self._train_step(state, batch)

                eval_count = 0
                if (step % self.config['hyperparams']['eval_freq'] == 0):
                    current_test = self._get_next_test_batch()
                    self._update_eval_rng()

                    metrics, prediction = self._evaluate(state.params, current_test)

                    self._save_output(self.save_dir, prediction, step)

                    eval_count += 1
                    if (eval_count % 1000 == 0) and (step != 0):
                        self._clear_result()

                    logging.info(
                        "step: {}, loss: {:.4f}, mse: {:.4f}".format(
                            step, metrics["loss"], metrics["mse"]
                        )
                    )

                if self.config["hyperparams"]["save_ckpt"]:
                    diffusor_mngr.save(
                        step,
                        args=ocp.args.Composite(
                            diffuser_state=ocp.args.StandardSave(state),
                            config=ocp.args.JsonSave(self.config.unfreeze()),
                        ),
                        metrics=float(metrics['loss'])
                    )

            diffusor_mngr.wait_until_finished()
