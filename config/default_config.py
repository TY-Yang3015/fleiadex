from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    config = ConfigDict()

    config.hyperparams = ConfigDict()
    config.hyperparams.learning_rate = 2e-4
    config.hyperparams.batch_size = 30
    config.hyperparams.epochs = 10000
    config.hyperparams.total_timesteps = 1000
    config.hyperparams.norm_groups = 8

    config.data_spec = ConfigDict()
    config.data_spec.image_size = 128
    config.data_spec.image_channels = 1
    config.data_spec.clip_min = -1.
    config.data_spec.clip_max = 1.
    config.data_spec.dataset_dir = '../../data/processed/Alaska/Sandwich'
    config.data_spec.validation_split = 0.2

    config.nn_spec = ConfigDict()
    config.nn_spec.first_conv_channels = 64
    config.nn_spec.channel_multiplier = [1, 2, 4, 8]
    config.nn_spec.widths = [config.nn_spec.first_conv_channels * mult for mult in config.nn_spec.channel_multiplier]
    config.nn_spec.has_attention = [False, False, True, True]
    config.nn_spec.num_res_blocks = 2

    config.latents = 2 * 768

    return config
