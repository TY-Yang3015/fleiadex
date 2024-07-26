from setuptools import setup, find_packages

VERSION = '0.0.1'

requirements = [
    'tensorflow',
    'jax',
    'jaxlib',
    'numpy',
    'pillow',
    'numpy',
    'hydra-core',
    'omegaconf',
    'optax',
    'flax',
    'einops'
]

setup(
    name='pleiades',
    version=VERSION,
    python_requires='>=3.10',
    liscense='Mozilla Public License Version 2.0',

    packages=find_packages(
        where="src",
        exclude=(
            'legacy',
            'notebooks',
            'literature',)
    ),
    package_dir={"": "src"},
    zip_safe=True,
    include_package_data=True,
    install_requires=requirements,
)
