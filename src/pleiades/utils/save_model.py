import logging

from flax.training import orbax_utils
import orbax
import os
import hydra
import shutil


def save_model(state, config, current_epoch):
    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    save_dir = str(os.path.join(hydra_dir, 'ckpt'))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ckpt = {'model': state, 'config': config}
    orbax_ckpter = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)

    if has_five_files(save_dir):
        clear_oldest_save(save_dir)

    save_dir += f'/epoch{int(current_epoch)}/'

    try:
        orbax_ckpter.save(save_dir, ckpt, save_args=save_args)
    except Exception:
        pass
    finally:
        save_dir = str(os.path.join(hydra_dir, 'ckpt'))
        os.rename(os.path.join(save_dir, os.listdir(save_dir)[-1]),
                  save_dir + f'/epoch{int(current_epoch)}/')


def clear_oldest_save(directory):
    oldest = os.path.join(directory, os.listdir(directory)[0])
    shutil.rmtree(oldest)

    logging.info("oldest ckpt removed.")


def has_five_files(directory):
    try:
        entries = os.listdir(directory)

        if len(entries) >= 5:
            return True
        else:
            return False

    except Exception as e:
        logging.info(f"an exception has occurred while counting ckpts: {e}")
        return False
