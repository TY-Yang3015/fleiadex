import hydra
from hydra.core.config_store import ConfigStore
import os, sys

from fleiadex.config import VAEConfig

sys.argv = sys.argv[:1]

cs = ConfigStore.instance()
cs.store(name="vae_config", node=VAEConfig)


@hydra.main(version_base=None, config_name="vae_config")
def execute_prediff_vae(config: VAEConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = config.global_config.use_which_gpus
    from fleiadex.trainers import get_vae_trainer

    trainer = get_vae_trainer(config)
    # trainer.load_vae_from("/home/arezy/Desktop/fleiadex/training_scripts/vae/outputs/"
    #                      "2024-08-03/22-07-34/results/vae_ckpt", load_config=False)
    trainer.train(auxiliary_metric=config.global_config.compute_auxiliary_metric)


def execute_vae(vae_type: str = 'prediff') -> None:
    if vae_type == 'prediff':
        execute_prediff_vae()
    else:
        raise NotImplementedError(f"variational autoencoder type {vae_type} not implemented.")
