from ml_collections import ConfigDict


def setup_config():
    config = ConfigDict()

    # Example of setting hyperparameters
    config.learning_rate = 0.001
    config.batch_size = 32
    config.epochs = 100

    # Model architecture parameters
    config.model = ConfigDict()
    config.model.hidden_units = [64, 128, 256]
    config.model.dropout_rate = 0.5

    # Data preprocessing parameters
    config.data = ConfigDict()
    config.data.shuffle_buffer_size = 10000
    config.data.validation_split = 0.2

    return config
