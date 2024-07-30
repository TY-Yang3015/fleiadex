# Project Pleiades

This is the developing latent diffusion module.

## Basic usage

The training scripts are contained in the `training_scripts` folder, where should contain two separate folders --- `diffusor` and `vae`. 
The entry functions are contained in the `main.py` files. For training the VAE and using it for the latent diffuser, just run `main.py`. 

As a feature of `hydra` (see Hydra doc [here](https://hydra.cc/docs/intro/)), as you run the script, all training results,
checkpoints and config settings will be saved in a separate (auto-created if not already) `outputs` folder.

The functions for independent usage of the VAE and diffuser will be updated soon.

## Setting up environment

You are advised to use `conda` as the environment manager. The key modules are

```shell
pip install --upgrade "jax[cuda12]" optax clu hydra-core tensorflow pillow einops flax
```

In principle, modules like `jaxlib` should come with `jax`/`optax`. This setup has been tested on linux (`ubuntu lts`), please
be advised to refer to the [jax installation page](https://jax.readthedocs.io/en/latest/installation.html) for further instruction
if you encountered any particular issue with the specific version of your OS. 

There are known issues with the auto-installed nvidia driver libs, especially some `cuda` components. Usually this can be solved by
updating `cuda`. 

## Change Hyperparameters

All config files are contained in the `config` module. Change the hyperparameters there can modify the most customisable 
features of the diffuser/vae. **Except the architecture of the backbone transformer**, see the `README.md` file in the
`src` folder. 

