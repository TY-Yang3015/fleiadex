import os
import glob
import tensorflow as tf
from PIL import Image
from tqdm.auto import tqdm
from datetime import datetime, timedelta


def load_image(path) -> list:
    return glob.glob(os.path.join(path, '*.jpg'))


def convert_name_to_time_label(name: str) -> str:
    year = int(name[:4])
    day_of_year = int(name[4:7])
    hour = int(name[7:9])
    minute = int(name[9:11])

    date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)

    standard_timestamp = datetime(date.year, date.month, date.day, hour, minute)

    return standard_timestamp.strftime('%Y-%m-%d_%H-%M-%S')


def crop_and_save_image(image_path: list, save_path: str) -> None:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pbar = tqdm(total=len(image_path))
    for i, image in enumerate(image_path):
        try:
            label = convert_name_to_time_label(image.split('/')[-1][:11])
            image = tf.io.read_file(image)
            image = tf.image.decode_image(image, channels=0, expand_animations=False).numpy()
            image = Image.fromarray(image[80:400, 80:400])
            # print(os.path.join(save_path, label + '.jpg'))
            image.save(os.path.join(save_path, label + '.jpg'))
        except ValueError as e:
            print(f"encountered ValueError while handling {image_path[i]}")
            print(f"error message: {str(e)}")
        pbar.update(1)
