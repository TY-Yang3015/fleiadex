from absl import logging
import omegaconf
from omegaconf import DictConfig, OmegaConf, open_dict
import numpy as np

class AutoDesigner:
    def __init__(self, config: DictConfig):
        self.config = config
        self.size = config.data_spec.image_size
        OmegaConf.set_readonly(self.config, False)

    def design_config(self):

        if self.config.nn_spec.nn_type == '2d_cnn':
            try:
                self.config.nn_spec.num_of_layers
                self.config.nn_spec.stride
                self.config.nn_spec.kernel_size
                self._design_2d_cnn_architecture()
                OmegaConf.set_readonly(self.config, True)
                return self.config

            except omegaconf.MissingMandatoryValue:
                self.config.nn_spec.num_of_layers = 4
                self.config.nn_spec.stride = 2
                self.config.nn_spec.kernel_size = 3
                base = int(self.size / 8)
                with open_dict(self.config):
                    self.config.nn_spec.features = [base * (2 ** i) for i
                            in reversed(range(1, self.config.nn_spec.num_of_layers + 1))]
                    self.config.nn_spec.max_feature = int(np.max(self.config.nn_spec.features))
                    n = self.config.data_spec.image_size / (self.config.nn_spec.stride
                                                            ** self.config.nn_spec.num_of_layers)
                    self.config.nn_spec.decoder_input = n * n * self.config.nn_spec.max_feature
                logging.warning("using default 2d cnn layers")
                OmegaConf.set_readonly(self.config, True)
                return self.config

        else:
            raise NotImplementedError('unknown nn architecture.')

    def _design_2d_cnn_architecture(self):
        stride = self.config.nn_spec.stride
        n_layers = self.config.nn_spec.num_of_layers
        base = int(self.size / (stride ** (n_layers - 1)))
        with open_dict(self.config):
            self.config.nn_spec.features = [base * (stride ** i) for i
                                            in reversed(range(self.config.nn_spec.num_of_layers))]
            self.config.nn_spec.max_feature = int(np.max(self.config.nn_spec.features))
            n = self.config.data_spec.image_size / (self.config.nn_spec.stride
                                                    ** self.config.nn_spec.num_of_layers)
            self.config.nn_spec.decoder_input = n * n * self.config.nn_spec.max_feature
