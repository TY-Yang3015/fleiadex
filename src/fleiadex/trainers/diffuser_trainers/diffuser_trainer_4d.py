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

import orbax.checkpoint as ocp
import etils.epath as path

import hydra
from omegaconf import OmegaConf

from fleiadex.nn_models import VAE
from fleiadex.utils import mse, TrainStateWithDropout, save_many_images
from fleiadex.diffuser import DDPMCore
from fleiadex.config import ConditionalDDPMConfig


class Trainer:
    def __init__(self, config: ConditionalDDPMConfig):

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

        # optimiser with exponential moving average and clipping
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
                + self.config["data_spec"]["sample_length"]
        )

        # load vae if not encoded
        if self.config['data_spec']['pre_encoded'] is False:
            if self.config['hyperparams']['load_vae_dir'] is not None:
                self.load_vae_from(self.config['hyperparams']['load_vae_dir'])
                self.__manual_load__ = True
            else:
                raise ValueError('for unencoded dataset, vae must be specified.')

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
                vae_config["hyperparams"]["batch_size"],
                vae_config["data_spec"]["image_size"],
                vae_config["data_spec"]["image_size"],
                vae_config["data_spec"]["image_channels"],
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

            expect = test_batch[..., self.config["data_spec"]["condition_length"]:]

            condition = test_batch[..., :self.config["data_spec"]["condition_length"]]
            decoded_test_batch = self.vae.apply_fn(
                {"params": params}, jnp.concatenate([condition, expect], axis=-1),
                method='decode'
            )

            decoded_condition = decoded_test_batch[..., :self.config["data_spec"]["condition_length"]]
            decoded_expect = decoded_test_batch[..., self.config["data_spec"]["condition_length"]:]

            noised_expect = diffuser.apply(
                {"params": params},
                expect, true, t,
                rngs={"dropout": self.dropout_rng},
                method="add_noise_to"
            )

            decoded_noised_expect = self.vae.apply_fn(
                {"params": self.vae.params}, jnp.concatenate([condition, noised_expect], axis=-1),
                method="decode"
            )[..., self.config["data_spec"]["condition_length"]:]

            denoised_expect = diffuser.apply(
                {"params": params},
                noised_expect, pred, t,
                self.eval_rng, rngs={"dropout": self.dropout_rng},
                method="denoise"
            )

            decoded_denoised_expect = self.vae.apply_fn(
                {"params": params}, jnp.concatenate([condition, denoised_expect], axis=-1),
                method="decode"
            )[..., self.config["data_spec"]["condition_length"]:]

            predictions = [decoded_condition, decoded_expect, decoded_noised_expect
                , decoded_denoised_expect]

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
                False,
                return_t=True,
                rngs={"dropout": self.dropout_rng},
            )

            metrics = compute_metric(pred, true)

            expect = test_batch[..., self.config["data_spec"]["condition_length"]:]
            noised_expect = diffuser.apply(
                {"params": params},
                expect, true, t,
                rngs={"dropout": self.dropout_rng},
                method="add_noise_to"
            )

            denoised_expect = diffuser.apply(
                {"params": params},
                noised_expect, pred, t,
                self.eval_rng, rngs={"dropout": self.dropout_rng},
                method="denoise"
            )

            predictions = [test_batch[..., : self.config["data_spec"]["condition_length"]],
                           expect, noised_expect, denoised_expect]

            return metrics, predictions

        return nn.apply(evaluate, self.diffuser)(
            {"params": params}, rngs={"dropout": self.dropout_rng}
        )

    def _save_output(self, save_dir, prediction, step, image_name='prediction'):
        if self.config["hyperparams"]["save_prediction"]:
            save_many_images(
                save_dir + "/predictions",
                prediction,
                f"/{image_name}_{int(step + 1)}.png",
                nrow=self.temporal_length * 2,
            )

    def _clear_result(self):
        directory = self.save_dir + "/predictions"
        shutil.rmtree(directory)
        os.makedirs(directory)

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

                if self.config['data_spec']['condition_length'] > 3:
                    logging.warning('only the first three condition channels will be visualised.')
            else:
                logging.warning("visualisation will be disabled.")

            logging.info("initializing model.")
            init_data = jnp.ones(
                (
                    self.config["hyperparams"]["batch_size"],
                    self.config["data_spec"]["output_image_size"],
                    self.config["data_spec"]["output_image_size"],
                    self.config["data_spec"]["image_channels"],
                ),
                jnp.float32,
            )

            rngs = {
                "params": random.PRNGKey(0),
                "dropout": random.PRNGKey(42),
            }
            collections = self.diffuser.init(rngs, init_data, self.train_rng, False)
            params = collections["params"]

            state = TrainStateWithDropout.create(
                apply_fn=self.diffuser.apply,
                params=params,
                tx=self.optimiser,
                key=self.dropout_rng,
            )

            sharding = jax.sharding.NamedSharding(
                mesh=jax.sharding.Mesh(jax.devices(), axis_names="model"),
                spec=jax.sharding.PartitionSpec(), )

            create_sharded_array = lambda x: jax.device_put(x, sharding)
            state = jax.tree_util.tree_map(create_sharded_array, state)

            save_path = ocp.test_utils.erase_and_create_empty(
                os.path.abspath(self.save_dir + "/ckpt")
            )
            save_options = ocp.CheckpointManagerOptions(
                max_to_keep=self.config["global_config"]["save_num_ckpts"],
                save_interval_steps=self.config["hyperparams"]["ckpt_freq"],
            )

            if self.config["hyperparams"]["save_ckpt"]:
                diffuser_mngr = ocp.CheckpointManager(
                    save_path,
                    options=save_options,
                    item_names=("diffuser_state", "config"),
                )

            for step in range(1, self.config["hyperparams"]["step"] + 1):

                batch = self._get_next_train_batch_pre_encoded()
                self._update_train_rng()

                state = self._train_step(state, batch)

                if self.config["hyperparams"]["save_ckpt"]:
                    diffuser_mngr.save(
                        step,
                        args=ocp.args.Composite(
                            diffuser_state=ocp.args.StandardSave(state),
                            config=ocp.args.JsonSave(self.config.unfreeze()),
                        ),
                    )

                current_test = self._get_next_test_batch_pre_coded()
                self._update_eval_rng()

                metrics, prediction = self._evaluate_pre_encoded(
                    state.params, current_test
                )

                logging.info(
                    "step: {}, loss: {:.4f}, mse: {:.4f}".format(
                        step + 1, metrics["loss"], metrics["mse"]
                    )
                )

                if force_visualisation:
                    self._save_output(self.save_dir, prediction, step, 'denoising')

                    if (step % 1000 == 0) and (step != 0):
                        self._clear_result()

            diffuser_mngr.wait_until_finished()

        else:
            if self.__vae_ready__ is False:
                raise ValueError(
                    "vae is not loaded. please run load_vae_from() method first."
                )

            logging.info("initializing model.")
            init_data = jnp.ones(
                (
                    self.config["hyperparams"]["batch_size"],
                    self.config["nn_spec"]["sample_input_shape"][1],
                    self.config["nn_spec"]["sample_input_shape"][2],
                    self.config["nn_spec"]["sample_input_shape"][3],
                ),
                jnp.float32,
            )

            rngs = {
                "params": random.PRNGKey(0),
                "dropout": random.PRNGKey(42),
            }
            collections = self.diffuser.init(rngs, init_data, self.train_rng, False)
            params = collections["params"]

            state = TrainStateWithDropout.create(
                apply_fn=self.diffuser.apply,
                params=params,
                tx=self.optimiser,
                key=self.dropout_rng,
            )

            if len(jax.devices()) == 0:
                sharding = jax.sharding.NamedSharding(
                    mesh=jax.sharding.Mesh(jax.devices(), axis_names="model"),
                    spec=jax.sharding.PartitionSpec(),
                )
            else:
                sharding = jax.sharding.NamedSharding(
                    mesh=jax.sharding.Mesh(jax.devices(), axis_names="model"),
                    spec=jax.sharding.PartitionSpec("model"),
                )
            create_sharded_array = lambda x: jax.device_put(x, sharding)
            state = jax.tree_util.tree_map(create_sharded_array, state)
            jax.tree_util.tree_map(lambda x: x.sharding, state)

            save_path = ocp.test_utils.erase_and_create_empty(
                os.path.abspath(self.save_dir + "/ckpt")
            )
            save_options = ocp.CheckpointManagerOptions(
                max_to_keep=5,
                save_interval_steps=self.config["hyperparams"]["ckpt_freq"],
            )

            if self.config["hyperparams"]["save_ckpt"]:
                diffuser_mngr = ocp.CheckpointManager(
                    save_path,
                    options=save_options,
                    item_names=("diffuser_state", "config"),
                )

            for step in range(1, self.config["hyperparams"]["step"] + 1):

                batch = self._get_next_train_batch()
                self._update_train_rng()

                state = self._train_step(state, batch)

                if self.config["hyperparams"]["save_ckpt"]:
                    diffuser_mngr.save(
                        step,
                        args=ocp.args.Composite(
                            diffuser_state=ocp.args.StandardSave(state),
                            config=ocp.args.JsonSave(self.config.unfreeze()),
                        ),
                    )

                current_test = self._get_next_test_batch()
                self._update_eval_rng()

                metrics, prediction = self._evaluate(state.params, current_test)

                # self._save_output(self.save_dir, prediction, step)

                if (step % 1000 == 0) and (step != 0):
                    self._clear_result()

                logging.info(
                    "step: {}, loss: {:.4f}, mse: {:.4f}".format(
                        step + 1, metrics["loss"], metrics["mse"]
                    )
                )

            diffuser_mngr.wait_until_finished()
