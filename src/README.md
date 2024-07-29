# Module Structure

This README file should give you a overview of the project structure. 

`pleiades` takes a modular design approach. All components are designated such that can be easily rewritten or replaced.


## `blocks` Folder

This folder contains all building blocks of the neural network, i.e. the `flax.linen.Module` subclasses. Most of the 
blocks come with documentation written in `restructuredText` template, compatible with `PyCharm`. 

## `diffuser` Folder

This folder currently only contains the [DDPM](https://arxiv.org/abs/2006.11239)-based diffusion interface. 
The training interface is included in `diffuser_trainer.py` file. `ddpm_core.py` and `ddpm_utils.py` contain 
the core interface and sampling manager.

## `errors` Folder

This is used for some unique error names. You can add custom errors here. 

## `transformer` Folder

This contains the [EarthFormer](https://arxiv.org/abs/2207.05833)-based UNet following the [PreDiff](https://arxiv.org/abs/2307.10422)
paper, which functions as the backbone of the latent diffusion model. **The `defualt_factory.py` file is the only place
where you can change the configuration of the `EarthFormer-UNet`**. 

## `utils` Folder

This folder contains all essential utility functions. The loss functions are maintained in the `loss.py` file.
Some custom `TrainState` for `BatchNorm` and `DropOut` are also defined here to prevent redundancy. 

## `vae` Folder

The entire VAE is built here, with the training interface included in the `vae_trainer.py` file.