import os
import requests
from bs4 import BeautifulSoup

# "https://cdn.star.nesdis.noaa.gov/GOES18/ABI/SECTOR/ak/Sandwich/" for Sandwich Data
# "https://cdn.star.nesdis.noaa.gov/GOES18/ABI/SECTOR/ak/GEOCOLOR/" for GeoColor Data

def fetch_data(base_url: str, save_folder: str):
    base_url = base_url

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    images = soup.find_all('a')
    for img in images:
        img_url = img.get('href')
        if img_url.endswith('500x500.jpg'):
            full_url = base_url + img_url
            img_data = requests.get(full_url).content
            with open(os.path.join(save_folder, img_url), 'wb') as handler:
                handler.write(img_data)
            print(f"downloaded {img_url}")

    print("all images have been downloaded.")
