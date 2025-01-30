# A simple setup.py file to build the package

from setuptools import setup, find_packages

setup(
    name='nanodiffusion',
    version='0.1',
    packages=find_packages(),
    install_requires=['torch'],
)
