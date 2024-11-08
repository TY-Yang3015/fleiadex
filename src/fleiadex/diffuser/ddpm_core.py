import flax.linen as nn
import jax.numpy as jnp
import jax
from flax.core import FrozenDict

from fleiadex.diffuser.ddpm_utils import DDPMManager
from fleiadex.nn_models.diffuser_backbones import *


class DDPMCore(nn.Module):
    config: FrozenDict
    diffusion_time_steps: int

    def setup(self) -> None:
        self.diffusion_mngr = DDPMManager(
            timestep=self.config['hyperparams']['diffusion_time_steps'],
            beta_start=1e-4, beta_end=0.02, clip_min=0, clip_max=1.0
        )

        self.__4d__ = False

        if self.config["global_config"]["use_diffuser_backbone"] == "earthformer":
            self.unet_backbone = EarthformerUNet(**self.config["nn_spec"])
        elif self.config["global_config"]["use_diffuser_backbone"] == "unet_5d_conv_2d":
            self.unet_backbone = UNet5DConv2D(**self.config["nn_spec"])
        elif self.config["global_config"]["use_diffuser_backbone"] == "unet_4d_conv_2d":
            self.unet_backbone = UNet4DConv2D(**self.config["nn_spec"])
            self.__4d__ = True
        elif self.config["global_config"]["use_diffuser_backbone"] == "unet_5d_conv_3d":
            raise NotImplementedError("this backbone is not implemented yet.")
        else:
            raise NotImplementedError(
                "only earthformer, vanilla2d and vanilla 3d unets are supported."
            )

    def __call__(self, x: jnp.ndarray, eval_key: jax.random.PRNGKey, train: bool,
                 return_t: bool = False):
        if self.__4d__:
            x_cond = x[..., :self.config["data_spec"]["condition_length"]]
            x_pred = x[..., self.config["data_spec"]["condition_length"]:]

            t = jax.random.randint(
                eval_key,
                shape=(x.shape[0], 1, 1, 1),
                minval=0,
                maxval=self.diffusion_time_steps,
                dtype=jnp.int32,
            )

            noise = jax.random.normal(eval_key, shape=x_pred.shape)

            noised_pred_t = self.diffusion_mngr.q_sample(x_pred, t, noise)

            t = t.reshape(t.shape[0])
            pred_noise = self.unet_backbone(noised_pred_t, x_cond, t, train)

            if return_t:
                return pred_noise, noise, t
            else:
                return pred_noise, noise

        else:
            x_cond = x[:, :self.config["data_spec"]["condition_length"]]
            x_pred = x[:, self.config["data_spec"]["condition_length"]:]

            t = jax.random.randint(
                eval_key,
                shape=(x.shape[0], 1, 1, 1, 1),
                minval=0,
                maxval=self.diffusion_time_steps,
                dtype=jnp.int32,
            )

            noise = jax.random.normal(eval_key, shape=x_pred.shape)

            noised_pred_t = self.diffusion_mngr.q_sample(x_pred, t, noise)

            pred_noise = self.unet_backbone(noised_pred_t, x_cond, t, train)

            if return_t:
                return pred_noise, noise, t
            else:
                return pred_noise, noise

    def add_noise_to(self, image, noise, t):
        if self.__4d__:
            t = t.reshape(t.shape[0], 1, 1, 1)
        noised_pred_t = self.diffusion_mngr.q_sample(image, t, noise)
        return noised_pred_t

    def denoise(self, noised_image, noise, t, eval_key):
        if self.__4d__:
            t = t.reshape(t.shape[0], 1, 1, 1)
        return self.diffusion_mngr.p_sample(noise, noised_image, t, eval_key, clip_denoised=False)

    def _generate_5d_prediction(self, x_cond, eval_key):

        if len(x_cond.shape) == 4:
            x_cond = jnp.expand_dims(x_cond, axis=0)

        z_t = jax.random.normal(
            eval_key,
            shape=(
                x_cond.shape[0],
                self.config["data_spec"]["prediction_length"],
                x_cond.shape[2],
                x_cond.shape[3],
                x_cond.shape[4],
            ),
        )

        for t in reversed(range(self.diffusion_time_steps)):
            ts = jnp.full((x_cond.shape[0], 1, 1, 1, 1), t, dtype=jnp.int32)
            pred_noise = self.unet_backbone(z_t, x_cond, ts, False)
            z_t = self.diffusion_mngr.p_sample(pred_noise, z_t, ts, clip_denoised=False)

        return z_t

    def _generate_4d_prediction(self, x_cond, eval_key):

        z_t = jax.random.normal(
            eval_key,
            shape=(
                x_cond.shape[0],
                x_cond.shape[1],
                x_cond.shape[2],
                self.config["data_spec"]["sample_length"],
            ),
        )

        for t in reversed(range(self.diffusion_time_steps)):
            ts = jnp.full((x_cond.shape[0], 1, 1, 1), t, dtype=jnp.int32)
            pred_noise = self.unet_backbone(z_t, x_cond, ts, False)
            z_t = self.diffusion_mngr.p_sample(pred_noise, z_t, ts, clip_denoised=False)

        return z_t

    def generate_prediction(self, x_cond, eval_key):
        if self.__4d__:
            return self._generate_4d_prediction(x_cond, eval_key)
        else:
            return self._generate_5d_prediction(x_cond, eval_key)

    def pred_noise(self, z_t, x_cond, ts):
        return self.unet_backbone(z_t, x_cond, ts, False)
