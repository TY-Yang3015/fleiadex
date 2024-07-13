import orbax
from flax.training import orbax_utils
from pleiades.vae.src.vae import VAE
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import flax.linen as nn
from flax.training import train_state
import os
import etils.epath as path

vae = VAE()
par = vae.init(jax.random.PRNGKey(0),
               jnp.zeros((15, 128, 128, 4)), jax.random.PRNGKey(1), train=False)

save_path = path.Path("outputs/2024-07-13/11-44-50/results/ckpt")
save_options = ocp.CheckpointManagerOptions(max_to_keep=5,
                                            save_interval_steps=2
                                            )
mngr = ocp.CheckpointManager(
    save_path, options=save_options
)




params = mngr.restore(9, args=ocp.args.StandardRestore(par['params']))



