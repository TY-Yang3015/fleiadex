import jax.numpy as jnp


def discriminator_loss(fake_judgement, origin_judgement):
    loss = -jnp.log(origin_judgement + 1e-6) - jnp.log(1 - fake_judgement + 1e-6)
    return jnp.minimum(jnp.mean(loss), 1000)
