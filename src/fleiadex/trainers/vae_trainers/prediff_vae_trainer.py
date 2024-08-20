import shutil
from absl import logging
from functools import partial
import os
from flax import linen as nn
from flax.core import FrozenDict
from jax import random, jit
import jax
from jax.lib import xla_bridge
import jax.numpy as jnp
import optax
from clu import platform
import tensorflow as tf
import hydra.utils as hydra_utils

import orbax.checkpoint as ocp
import etils.epath as path

import hydra
from omegaconf import OmegaConf

from fleiadex.nn_models import VAE
from fleiadex.utils import (
    mse,
    kl_divergence,
    ssim,
    discriminator_loss,
    TrainStateWithDropout,
    TrainStateWithBatchStats,
    save_image
)
from fleiadex.nn_models import Discriminator

from fleiadex.config import VAEConfig


class Trainer:
    def __init__(self, config: VAEConfig):
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
        latent_rng, self.eval_rng, self.dropout_rng, self.train_key = self._init_rng()
        # train_key is only used to split keys
        self.train_key, self.train_rng = random.split(self.train_key)

        # calculate latent size
        latent_size = 1  # initialise
        for down_factor in self.config["nn_spec"][
            "encoder_spatial_downsample_schedule"
        ]:
            latent_size *= down_factor
        latent_size = int(self.config["data_spec"]["image_size"] / latent_size)

        # generate random latent noise for sample generation
        self.latent_sample = random.normal(
            latent_rng,
            (
                self.config["hyperparams"]["sample_size"],
                latent_size,
                latent_size,
                self.config["nn_spec"]["decoder_latent_channels"],
            ),
        )

        # initialise dataset with custom pipelines
        logging.info("initializing dataset.")
        self.data_loader = hydra_utils.instantiate(config.data_spec, _recursive_=False)

        (
            self.train_ds,
            self.test_ds,
        ) = self.data_loader.write_data_summary().get_train_test_dataset()

        # initialise the save directory
        self.save_dir = self._init_savedir()

        # instantiate VAE
        self.vae = VAE(**self.config["nn_spec"])

        # get optimisers
        self.vae_optimiser = self._get_optimiser(self.config)
        self.disc_optimiser = self._get_optimiser(self.config)

        self.vae_optimiser = optax.chain(
            optax.clip_by_global_norm(self.config['hyperparams']['gradient_clipping']),
            self.vae_optimiser,)
        self.disc_optimiser = optax.chain(
            optax.clip_by_global_norm(self.config['hyperparams']['gradient_clipping']),
            self.disc_optimiser,)

        # instantiate discriminator
        self.discriminator = Discriminator()

        # strict private variable to record whether a manual loading from ckpts was used.
        self.__manual_load__ = False
        self.restored_vae_state = None
        self.restored_discriminator_state = None

        if self.config['hyperparams']['load_ckpt_dir'] is not None:
            self.load_vae_from(self.config['hyperparams']['load_ckpt_dir'],
                               self.config['hyperparams']['load_config'],
                               self.config['hyperparams']['ckpt_step'])
            self.__manual_load__ = True

    def _init_rng(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        init_key = random.PRNGKey(42)
        latent_rng, eval_rng, dropout_rng, train_key = random.split(init_key, 4)
        return latent_rng, eval_rng, dropout_rng, train_key

    def _init_savedir(self) -> str:
        save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        save_dir = str(os.path.join(save_dir, "results"))
        os.makedirs(save_dir)
        os.makedirs(save_dir + "/sample")
        os.makedirs(save_dir + "/comparison")
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

    @partial(jit, static_argnames="self")
    def _vae_init_step(self, vae_state, batch):
        def vae_loss_fn(params):
            recon_x, mean, logvar = vae_state.apply_fn(
                {"params": params},
                batch,
                self.train_rng,
                True,
                rngs={"dropout": self.dropout_rng},
            )

            mse_loss = mse(recon_x, batch).mean()
            kld_loss = kl_divergence(mean, logvar).mean()
            vae_loss = mse_loss + kld_loss * self.config["hyperparams"]["kld_weight"]

            return vae_loss

        grads = jax.grad(vae_loss_fn)(vae_state.params)
        vae_state = vae_state.apply_gradients(grads=grads)

        return vae_state

    @partial(jit, static_argnames="self")
    def _vae_train_step(self, vae_state, discriminator_state, batch):
        def vae_loss_fn(params):
            recon_x, mean, logvar = vae_state.apply_fn(
                {"params": params},
                batch,
                self.train_rng,
                True,
                rngs={"dropout": self.dropout_rng},
            )

            mse_loss = mse(recon_x, batch).mean()
            kld_loss = kl_divergence(mean, logvar).mean()

            vae_loss = mse_loss + kld_loss * self.config["hyperparams"]["kld_weight"]

            fake_judgement = discriminator_state.apply_fn(
                {
                    "params": discriminator_state.params,
                    "batch_stats": discriminator_state.batch_stats,
                },
                recon_x,
                False,
                mutable=["batch_stats"],
            )[0]
            origin_judgement = discriminator_state.apply_fn(
                {
                    "params": discriminator_state.params,
                    "batch_stats": discriminator_state.batch_stats,
                },
                batch,
                False,
                mutable=["batch_stats"],
            )[0]

            disc_loss = discriminator_loss(fake_judgement, origin_judgement)
            disc_loss *= -self.config["hyperparams"]["disc_weight"]

            vae_loss += disc_loss

            return vae_loss

        grads = jax.grad(vae_loss_fn)(vae_state.params)
        vae_state = vae_state.apply_gradients(grads=grads)

        return vae_state

    @partial(jit, static_argnames="self")
    def _discriminator_train_step(self, discriminator_state, vae_state, batch):
        def discriminator_loss_fn(params):
            origin_judgement = discriminator_state.apply_fn(
                {"params": params, "batch_stats": discriminator_state.batch_stats},
                batch,
                True,
                mutable=["batch_stats"],
            )[0]

            fake = vae_state.apply_fn(
                {"params": vae_state.params},
                batch,
                self.eval_rng,
                False,
            )[0]

            fake_judgement = discriminator_state.apply_fn(
                {"params": params, "batch_stats": discriminator_state.batch_stats},
                fake,
                True,
                mutable=["batch_stats"],
            )[0]

            discriminator_loss_val = discriminator_loss(
                fake_judgement, origin_judgement
            )

            return discriminator_loss_val

        grads = jax.grad(discriminator_loss_fn)(discriminator_state.params)
        discriminator_state = discriminator_state.apply_gradients(grads=grads)

        return discriminator_state

    @partial(jit, static_argnames="self")
    def _evaluate_vae(
        self, params, images, latent_sample, eval_rng, discriminator_state
    ):
        def compute_vae_metric(
            recon_x: jnp.ndarray, x: jnp.ndarray, mean: jnp.ndarray, logvar: jnp.ndarray
        ) -> dict[str, jnp.ndarray]:
            fake_judgement = discriminator_state.apply_fn(
                {
                    "params": discriminator_state.params,
                    "batch_stats": discriminator_state.batch_stats,
                },
                recon_x,
                False,
                mutable=["batch_stats"],
            )[0]
            origin_judgement = discriminator_state.apply_fn(
                {
                    "params": discriminator_state.params,
                    "batch_stats": discriminator_state.batch_stats,
                },
                images,
                False,
                mutable=["batch_stats"],
            )[0]

            disc_loss = discriminator_loss(fake_judgement, origin_judgement)
            disc_loss = disc_loss * self.config["hyperparams"]["disc_weight"]
            mse_loss = mse(recon_x, x).mean()
            kld_loss = (
                kl_divergence(mean, logvar).mean()
                * self.config["hyperparams"]["kld_weight"]
            )
            return {
                "mse": mse_loss,
                "kld": kld_loss,
                "disc_loss": disc_loss,
                "loss": disc_loss + mse_loss + kld_loss,
            }

        def evaluate_vae(vae, train):
            size = self.config["data_spec"]["image_size"]
            channels = self.config["data_spec"]["image_channels"]

            recon_images, mean, logvar = vae(images, eval_rng, train)
            comparison = jnp.concatenate(
                [
                    images[:].reshape(-1, size, size, channels),
                    recon_images[:].reshape(
                        recon_images.shape[0], size, size, channels
                    ),
                ]
            )

            generate_images = vae.generate(latent_sample)
            generate_images = generate_images.reshape(-1, size, size, channels)

            metrics = compute_vae_metric(recon_images, images, mean, logvar)
            return metrics, comparison, generate_images

        return nn.apply(evaluate_vae, self.vae)(
            {"params": params}, False, rngs={"dropout": self.dropout_rng}
        )

    @partial(jit, static_argnames="self")
    def _evaluate_init_vae(self, params, images, latent_sample, eval_rng):
        def compute_vae_metric(
            recon_x: jnp.ndarray, x: jnp.ndarray, mean: jnp.ndarray, logvar: jnp.ndarray
        ) -> dict[str, jnp.ndarray]:
            mse_loss = mse(recon_x, x).mean()
            kld_loss = (
                kl_divergence(mean, logvar).mean()
                * self.config["hyperparams"]["kld_weight"]
            )
            return {
                "mse": mse_loss,
                "kld": kld_loss,
                "disc_loss": None,
                "loss": (mse_loss + kld_loss),
            }

        def evaluate_vae(vae, train):
            size = self.config["data_spec"]["image_size"]
            channels = self.config["data_spec"]["image_channels"]

            recon_images, mean, logvar = vae(images, eval_rng, train)
            comparison = jnp.concatenate(
                [
                    images[:].reshape(-1, size, size, channels),
                    recon_images[:].reshape(
                        recon_images.shape[0], size, size, channels
                    ),
                ]
            )

            generate_images = vae.generate(latent_sample)
            generate_images = generate_images.reshape(-1, size, size, channels)

            metrics = compute_vae_metric(recon_images, images, mean, logvar)
            return metrics, comparison, generate_images

        return nn.apply(evaluate_vae, self.vae)(
            {"params": params}, False, rngs={"dropout": self.dropout_rng}
        )

    @partial(jit, static_argnames="self")
    def _evaluate_discriminator(self, params, batch_stats, vae_state, images):
        def compute_discriminator_metric(discriminator) -> dict[str, jnp.ndarray]:
            origin_judgement = discriminator(images, False)
            fake = vae_state.apply_fn(
                {"params": vae_state.params},
                images,
                self.eval_rng,
                False,
            )[0]
            fake_judgement = discriminator(fake, False)
            discriminator_loss_val = discriminator_loss(
                fake_judgement, origin_judgement
            )
            return {"loss": discriminator_loss_val}

        return nn.apply(compute_discriminator_metric, self.discriminator)(
            {"params": params, "batch_stats": batch_stats}
        )

    def _save_output(self, save_dir, comparison, sample, epoch):

        if self.config["hyperparams"]["save_comparison"]:
            save_image(
                save_dir + "/comparison",
                comparison,
                f"/comparison_{int(epoch + 1)}.png",
                nrow=self.config["hyperparams"]["batch_size"],
            )

        if self.config["hyperparams"]["save_sample"]:
            save_image(
                save_dir + "/sample",
                sample,
                f"/sample_{int(epoch + 1)}.png",
                nrow=self.config["hyperparams"]["sample_size"]
                if self.config["hyperparams"]["sample_size"] <= 10
                else 10,
            )

    def _clear_result(self):
        directory = self.save_dir + "/comparison"
        shutil.rmtree(directory)
        os.makedirs(directory)
        directory = self.save_dir + "/sample"
        shutil.rmtree(directory)
        os.makedirs(directory)

    def load_vae_from(self, ckpt_dir, load_config=True, step=None) -> None:
        if self.__manual_load__ is True:
            logging.warning('ckpt cannot be loaded twice. this call will be ignored.')
            return None

        logging.info("loading vae from %s", ckpt_dir)

        if isinstance(ckpt_dir, str):
            disc_save_dir = path.Path(
                ckpt_dir.strip(ckpt_dir.split("/")[-1]) + "disc_ckpt"
            )
            ckpt_dir = path.Path(ckpt_dir)
        else:
            raise ValueError("ckpt dir must be str.")

        mngr = ocp.CheckpointManager(ckpt_dir, item_names=("vae_state", "config"))

        if load_config:
            if step:
                restored_config = mngr.restore(
                    step, args=ocp.args.Composite(config=ocp.args.JsonRestore())
                )
            else:
                restored_config = mngr.restore(
                    mngr.latest_step(),
                    args=ocp.args.Composite(config=ocp.args.JsonRestore()),
                )

            self.config = FrozenDict(restored_config["config"])
        else:
            logging.warning("not loading config may lead to unexpected behaviour.")

        self.vae = VAE(**self.config["nn_spec"])

        init_data = jnp.ones(
            (
                self.config["hyperparams"]["batch_size"],
                self.config["data_spec"]["image_size"],
                self.config["data_spec"]["image_size"],
                self.config["data_spec"]["image_channels"],
            ),
            jnp.float32,
        )

        rngs = {"params": random.PRNGKey(0), "dropout": random.PRNGKey(42)}
        params = self.vae.init(rngs, init_data, random.key(0), False)["params"]

        self.vae_optimiser = self._get_optimiser(self.config)
        self.disc_optimiser = self._get_optimiser(self.config)

        self.vae_optimiser = optax.chain(
            optax.clip_by_global_norm(self.config['hyperparams']['gradient_clipping']),
            self.vae_optimiser, )

        self.disc_optimiser = optax.chain(
            optax.clip_by_global_norm(self.config['hyperparams']['gradient_clipping']),
            self.disc_optimiser, )

        vae_state = TrainStateWithDropout.create(
            apply_fn=self.vae.apply,
            params=params,
            tx=self.vae_optimiser,
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

        if step:
            restored = mngr.restore(
                step,
                args=ocp.args.Composite(vae_state=ocp.args.StandardRestore(vae_state)),
            )["vae_state"]
        else:
            restored = mngr.restore(
                mngr.latest_step(),
                args=ocp.args.Composite(vae_state=ocp.args.StandardRestore(vae_state)),
            )["vae_state"]
            if self.config["hyperparams"]["save_discriminator"]:
                mngr = ocp.CheckpointManager(disc_save_dir)
                discriminator_vars = self.discriminator.init(rngs, init_data, False)
                discriminator_params, discriminator_batch_stats = (
                    discriminator_vars["params"],
                    discriminator_vars["batch_stats"],
                )

                discriminator_state = TrainStateWithBatchStats.create(
                    apply_fn=self.discriminator.apply,
                    params=discriminator_params,
                    tx=self.disc_optimiser,
                    batch_stats=discriminator_batch_stats,
                )

                discriminator_state = jax.tree_util.tree_map(create_sharded_array, discriminator_state)
                discriminator_state = jax.tree_util.tree_map(
                    ocp.utils.to_shape_dtype_struct, discriminator_state
                )

                if step:
                    restored_disc = mngr.restore(
                        step, args=ocp.args.StandardRestore(discriminator_state)
                    )
                else:
                    restored_disc = mngr.restore(
                        mngr.latest_step(),
                        args=ocp.args.StandardRestore(discriminator_state),
                    )

        logging.info("loading succeeded.")

        self.__manual_load__ = True

        self.restored_vae_state = restored
        if self.config["hyperparams"]["save_discriminator"]:
            self.restored_discriminator_state = restored_disc

        del mngr

    def train(self):
        logging.info("initializing model.")
        init_data = jnp.ones(
            (
                self.config["hyperparams"]["batch_size"],
                self.config["data_spec"]["image_size"],
                self.config["data_spec"]["image_size"],
                self.config["data_spec"]["image_channels"],
            ),
            jnp.float32,
        )

        rngs = {"params": random.PRNGKey(0), "dropout": random.PRNGKey(42)}
        params = self.vae.init(rngs, init_data, random.key(0), False)["params"]

        if self.__manual_load__:
            logging.info("attempting to create vae training state from ckpt...")
            vae_state = self.restored_vae_state
            logging.info("success.")
        else:
            vae_state = TrainStateWithDropout.create(
                apply_fn=self.vae.apply,
                params=params,
                tx=self.vae_optimiser,
                key=self.dropout_rng,
            )

        if self.__manual_load__ and self.config["hyperparams"]["save_discriminator"]:
            logging.info(
                "attempting to create discriminator training state from ckpt..."
            )
            discriminator_state = self.restored_discriminator_state
            logging.info("success.")
        else:
            discriminator_vars = self.discriminator.init(rngs, init_data, False)
            discriminator_params, discriminator_batch_stats = (
                discriminator_vars["params"],
                discriminator_vars["batch_stats"],
            )

            discriminator_state = TrainStateWithBatchStats.create(
                apply_fn=self.discriminator.apply,
                params=discriminator_params,
                tx=self.disc_optimiser,
                batch_stats=discriminator_batch_stats,
            )

        sharding = jax.sharding.NamedSharding(
            mesh=jax.sharding.Mesh(jax.devices(), axis_names="model"),
            spec=jax.sharding.PartitionSpec(),
        )

        create_sharded_array = lambda x: jax.device_put(x, sharding)
        vae_state = jax.tree_util.tree_map(create_sharded_array, vae_state)
        discriminator_state = jax.tree_util.tree_map(
            create_sharded_array, discriminator_state
        )

        save_vae_path = ocp.test_utils.erase_and_create_empty(
            os.path.abspath(self.save_dir + "/vae_ckpt")
        )
        save_disc_path = ocp.test_utils.erase_and_create_empty(
            os.path.abspath(self.save_dir + "/disc_ckpt")
        )
        save_options = ocp.CheckpointManagerOptions(
            max_to_keep=self.config["global_config"]["save_num_ckpts"],
            save_interval_steps=self.config["hyperparams"]["ckpt_freq"],
            best_fn=lambda metrics: float(metrics),
            best_mode='min'
        )

        if self.config["hyperparams"]["save_ckpt"]:
            vae_mngr = ocp.CheckpointManager(
                save_vae_path, options=save_options, item_names=("vae_state", "config")
            )
            if self.config["hyperparams"]["save_discriminator"]:
                disc_mngr = ocp.CheckpointManager(save_disc_path, options=save_options)

        for step in range(self.config["hyperparams"]["step"] + 1):

            batch = next(self.train_ds)
            if len(batch) != self.config["hyperparams"]["batch_size"]:
                batch = next(self.train_ds)
            self._update_train_rng()

            if step > self.config["hyperparams"]["discriminator_start_after"]:
                vae_state = self._vae_train_step(vae_state, discriminator_state, batch)
                discriminator_state = self._discriminator_train_step(
                    discriminator_state, vae_state, batch
                )
            else:
                vae_state = self._vae_init_step(vae_state, batch)

            eval_count = 0
            if (step % self.config['hyperparams']['eval_freq'] == 0):
                current_test_ds = next(self.test_ds)
                if len(current_test_ds) != self.config["hyperparams"]["batch_size"]:
                    current_test_ds = next(self.test_ds)

                if step > self.config["hyperparams"]["discriminator_start_after"]:
                    vae_metrics, comparison, sample = self._evaluate_vae(
                        vae_state.params,
                        current_test_ds,
                        self.latent_sample,
                        self.eval_rng,
                        discriminator_state,
                    )
                    discriminator_metric = self._evaluate_discriminator(
                        discriminator_state.params,
                        discriminator_state.batch_stats,
                        vae_state,
                        current_test_ds,
                    )
                else:
                    vae_metrics, comparison, sample = self._evaluate_init_vae(
                        vae_state.params, current_test_ds, self.latent_sample, self.eval_rng
                    )

                self._save_output(self.save_dir, comparison.copy(), sample.copy(), step)

                eval_count += 1
                if (eval_count % 1000 == 0) and (step != 0):
                    self._clear_result()

                if step > self.config["hyperparams"]["discriminator_start_after"]:
                    logging.info(
                        "step: {}, loss: {:.4f}, mse: {:.4f}, kld: {:.4f}, disc: {:.4f}".format(
                            step,
                            vae_metrics["loss"],
                            vae_metrics["mse"],
                            vae_metrics["kld"],
                            vae_metrics["disc_loss"],
                        )
                    )

                    if jnp.isnan(vae_metrics["loss"]):
                        logging.warning("nan data detected. auto-break.")
                        break

                    logging.info(
                        "discriminator loss: {:.4f}".format(discriminator_metric["loss"])
                    )
                else:
                    logging.info(
                        "step: {}, loss: {:.4f}, mse: {:.4f}, kld: {:.4f}".format(
                            step,
                            vae_metrics["loss"],
                            vae_metrics["mse"],
                            vae_metrics["kld"],
                        )
                    )

                    if jnp.isnan(vae_metrics["loss"]):
                        logging.warning("nan data detected. auto-break.")
                        break

                if self.config['global_config']["compute_auxiliary_metric"]:
                    batchwise_ssim = ssim(
                        comparison[: self.config["hyperparams"]["batch_size"]],
                        comparison[self.config["hyperparams"]["batch_size"] :],
                    )
                    logging.info("auxiliary SSIM: {:.4f}".format(batchwise_ssim))

            if self.config["hyperparams"]["save_ckpt"]:
                vae_mngr.save(
                    step,
                    args=ocp.args.Composite(
                        vae_state=ocp.args.StandardSave(vae_state),
                        config=ocp.args.JsonSave(self.config.unfreeze()),

                    ),
                    metrics=float(vae_metrics['loss'])
                )

                if self.config["hyperparams"]["save_discriminator"]:
                    disc_mngr.save(
                        step, args=ocp.args.StandardSave(discriminator_state),
                        metrics=float(vae_metrics['loss'])
                    )

        vae_mngr.wait_until_finished()
