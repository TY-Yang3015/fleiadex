from fleiadex.config import LDMConfig, ConditionalDDPMConfig


def get_diffuser_trainer(config: LDMConfig | ConditionalDDPMConfig):
    if config.global_config.use_diffuser_backbone == "earthformer":
        if config.nn_spec.use_relative_position:
            from fleiadex.trainers.diffuser_trainers.diffuser_trainer_with_constants import (
                Trainer,
            )

            return Trainer(config)
        else:
            from fleiadex.trainers.diffuser_trainers.diffuser_trainer_no_constant import (
                Trainer,
            )

            return Trainer(config)
    elif config.global_config.use_diffuser_backbone.split("_")[1] == "4d":
        from fleiadex.trainers.diffuser_trainers.diffuser_trainer_4d import (
            Trainer,
        )

        return Trainer(config)
    else:
        from fleiadex.trainers.diffuser_trainers.diffuser_trainer_no_constant import (
            Trainer,
        )

        return Trainer(config)
