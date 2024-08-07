from config.vae_config import VAEConfig
from src.pleiades.trainers.vae_trainers.prediff_vae_trainer import Trainer


def get_vae_trainer(config: VAEConfig):
    return Trainer(config)
