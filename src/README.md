# Module Structure of `pleiades`

This README file should give you a overview of the project structure. 

`pleiades` takes a modular design approach. All components are designated such that can be easily rewritten or replaced.


## `blocks` Folder

This folder contains all building blocks of the neural network, i.e. the `flax.linen.Module` subclasses. Most of the 
blocks come with documentation written in `restructuredText` template, compatible with `PyCharm`. 

## `data_module` Folder

This is the data-loading module. `.npy` files should be handled by the `DataLoader` class. The `thunderstorm_dataloader`
submodule contains the designated data loader for thunderstorm dataset.

More features will be added soon.

## `diffuser` Folder

This folder currently only contains the [DDPM](https://arxiv.org/abs/2006.11239)-based diffusion interface. 
The training interface is included in `diffuser_trainer.py` file. `ddpm_core.py` and `ddpm_utils.py` contain 
the core interface and sampling manager.

## `errors` Folder

This is used for some unique error names. You can add custom errors here. 

## `nn_models` Folder

- ### `diffuser_backbones` Subfolder

This contains the [EarthFormer](https://arxiv.org/abs/2207.05833)-based UNet following the [PreDiff](https://arxiv.org/abs/2307.10422)
paper, which functions as the backbone of the latent diffusion model. **The `defualt_factory.py` file is the only place
where you can change the configuration of the `EarthFormer-UNet`**. The `vanilla_unet2d.py` file contains a 2d u-net with
attentions in all stages.

- ### `predictor` Subfolder

This contains the predictor for binary map of the thunderstorm prediction and radar signal.

- ### `vae` Subfolder

This contains the variational autoencoder with a adversarial (GAN) loss.

## `trainers` Folder

All trainers for three components are stored here. **One should access the trainers only with `get_xxxxx_trainer`
functions.**

## `utils` Folder

This folder contains all essential utility functions. The loss functions are maintained in the `loss_lib` submodule.
Custom `TrainState`s are stored in `train_states` submodule. The metric functions are contained in the `metric_lib`
submodule. 
