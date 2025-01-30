# A simple setup.py file to build the package

from setuptools import setup

setup(
    name='nanodiffusion',
    version='0.1',
    description='A minimalistic library for training diffusion and flow matching models',
    packages=['nanodiffusion'],
    package_dir={'': 'src'},
    install_requires=['torch'],
)
