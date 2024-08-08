from config.ldm_config import LDMConfig


def get_diffuser_trainer(config: LDMConfig):
    if config.global_config.use_diffuser_backbone == 'earthformer':
        if config.nn_spec.use_relative_position:
            from src.pleiades.trainers.diffuser_trainers.diffuser_trainer_with_constants import Trainer
            return Trainer(config)
        else:
            from src.pleiades.trainers.diffuser_trainers.diffuser_trainer_no_constant import Trainer
            return Trainer(config)
    else:
        from src.pleiades.trainers.diffuser_trainers.diffuser_trainer_no_constant import Trainer
        return Trainer(config)