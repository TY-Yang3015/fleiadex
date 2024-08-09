import flax.linen as nn
import jax.numpy as jnp
import jax
from flax.core import FrozenDict

from src.pleiades.diffuser.ddpm_utils import DDPMManager
from src.pleiades.nn_models.diffuser_backbones.earthformer_unet import EarthformerUNet
from src.pleiades.nn_models.diffuser_backbones import VanillaUNet2D


class DDPMCore(nn.Module):
    config: FrozenDict
    diffusion_time_steps: int

    def setup(self) -> None:
        self.diffusion_mngr = DDPMManager(
            timestep=1000, beta_start=1e-4, beta_end=0.02, clip_min=0, clip_max=1.0
        )

        if self.config["global_config"]["use_diffuser_backbone"] == "earthformer":
            self.unet_backbone = EarthformerUNet(**self.config["nn_spec"])
        elif self.config["global_config"]["use_diffuser_backbone"] == "vanilla2d":
            self.unet_backbone = VanillaUNet2D(**self.config["nn_spec"])
        elif self.config["global_config"]["use_diffuser_backbone"] == "vanilla3d":
            pass
        else:
            raise NotImplementedError(
                "only earthformer, vanilla2d and vanilla 3d unets are supported."
            )

    def __call__(self, x, eval_key, train):
        x_cond = x[:, : self.config["data_spec"]["condition_length"]]
        x_pred = x[:, self.config["data_spec"]["condition_length"] :]

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

        return pred_noise, noise

    def generate_prediction(self, x_cond, eval_key):

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
