import jax.numpy as jnp


def discriminator_loss(fake_judgement, origin_judgement):
    loss = -jnp.log(origin_judgement) - jnp.log(1 - fake_judgement)
    return jnp.minimum(jnp.mean(loss), 1000)
