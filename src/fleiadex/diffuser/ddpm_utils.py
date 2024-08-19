import flax.linen as nn
import jax.numpy as jnp
import jax


class DDPMManager(nn.Module):
    timestep: int = (1000,)
    beta_start: float = (1e-4,)
    beta_end: float = (0.02,)
    clip_min: float = (0,)
    clip_max: float = 1.0

    def setup(self) -> None:
        self.betas = jnp.linspace(self.beta_start, self.beta_end, self.timestep)

        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = jnp.append(1.0, self.alpha_cumprod[:-1])

        self.sqrt_alphas_cumprod = jnp.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alpha_cumprod)

        self.log_one_minus_alphas_cumprod = jnp.log(1.0 - self.alpha_cumprod)
        self.sqrt_recip_alphas_cumprod = jnp.sqrt(1 / self.alpha_cumprod)
        self.sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / self.alpha_cumprod - 1.0)

        self.posterior_var = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alpha_cumprod)
        )

        self.posterior_log_variance_clipped = jnp.log(
            jnp.maximum(self.posterior_var, 1e-20)
        )

        self.posterior_mean_coef1 = (
            self.betas * jnp.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alpha_cumprod)
        )

        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * jnp.sqrt(self.alphas)
            / (1.0 - self.alpha_cumprod)
        )

    def q_mean_var(self, x_start, t):
        mean = self.sqrt_alphas_cumprod[t] * x_start
        var = 1.0 - self.alpha_cumprod[t]
        log_var = self.log_one_minus_alphas_cumprod[t]

        return mean, var, log_var

    def q_sample(self, x_start, t, noise):
        return (
            self.sqrt_alphas_cumprod[t] * x_start
            + self.sqrt_one_minus_alphas_cumprod[t] * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self.sqrt_recip_alphas_cumprod[t] * x_t
            - self.sqrt_recipm1_alphas_cumprod[t] * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        )

        posterior_var = self.posterior_var[t]
        posterior_log_var = self.posterior_log_variance_clipped[t]

        return posterior_mean, posterior_var, posterior_log_var

    def p_mean_var(self, pred_noise, x, t, clip_denoised=False):
        x_recon = self.predict_start_from_noise(x, t, pred_noise)
        if clip_denoised:
            x_recon = jnp.clip(x_recon, self.clip_min, self.clip_max)

        model_mean, posterior_var, posterior_log_var = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_var, posterior_log_var

    def p_sample(self, pred_noise, x, t, eval_key, clip_denoised=False):
        model_mean, _, model_log_var = self.p_mean_var(pred_noise, x, t, clip_denoised)

        noise = jax.random.normal(eval_key, x.shape)

        nonzero_mask = 1 - jnp.equal(t, 0)

        return model_mean + nonzero_mask * jnp.exp(0.5 * model_log_var) * noise
