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

save_path = path.Path("/home/arezy/Desktop/ProjectPleiades/training_scripts/vae/outputs/2024-07-19/14-32-00/results/ckpt")
save_options = ocp.CheckpointManagerOptions(max_to_keep=5,
                                            save_interval_steps=2
                                            )
mngr = ocp.CheckpointManager(
    save_path, options=save_options
)




params = mngr.restore(10, args=ocp.args.StandardRestore(par['params']))



