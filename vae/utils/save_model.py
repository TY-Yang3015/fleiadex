from flax.training import orbax_utils
import orbax
import os
import hydra


def save_model(state, config, current_epoch):
    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    save_dir = str(os.path.join(save_dir, 'ckpt'))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ckpt = {'model': state, 'config': config}
    orbax_ckpter = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)

    if has_more_than_five_files(save_dir):
        clear_save(save_dir)

    save_dir += '/epoch' + str(current_epoch)

    orbax_ckpter.save(save_dir, ckpt, save_args=save_args)


def clear_save(path):
    # Walk through the directory in reverse order to ensure all files and subdirectories are deleted
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))  # Delete the file
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))   # Delete the subdirectory


def has_more_than_five_files(directory):
    try:
        # Get the list of entries in the directory
        entries = os.listdir(directory)

        # Initialize a counter for files
        file_count = 0

        # Loop through the entries and count the files
        for entry in entries:
            entry_path = os.path.join(directory, entry)
            if os.path.isfile(entry_path):
                file_count += 1
                # If the count exceeds 10, return True
                if file_count > 5:
                    return True

        # Return False if the count is 10 or less
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


