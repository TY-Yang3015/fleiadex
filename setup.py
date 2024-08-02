from setuptools import setup, find_packages

VERSION = '0.0.1'

requirements = [
    'tensorflow',
    'jax[cuda12]',  # change this if you wish to run on CPU/TPU.
    'jaxlib',
    'pillow',
    'hydra-core',
    'optax',
    'flax',
    'einops',
    'clu'
]

setup(
    name='pleiades',
    version=VERSION,
    python_requires='>=3.10',
    liscense='Mozilla Public License Version 2.0',

    packages=find_packages(
        where="prediff_vae",
        exclude=(
            'legacy',
            'notebooks',
            'literature',)
    ),
    package_dir={"": "prediff_vae"},
    zip_safe=True,
    include_package_data=True,
    install_requires=requirements,
)
