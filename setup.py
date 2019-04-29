#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup

Module installs the annom package
Can be run via command: python setup.py install (or develop)

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Mar 11, 2019
"""

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

version = '0.0.1'

args = dict(
    name='annom',
    version=version,
    description="pytorch-based anomaly detection in synthesis",
    long_description=readme,
    author='Jacob Reinhold',
    author_email='jacob.reinhold@jhu.edu',
    url='https://gitlab.com/jcreinhold/annom',
    packages=find_packages(exclude=('tests')),
)

setup(install_requires=['numpy>=1.15.4',
                        'scipy',
                        'torch>=1.0.0',
                        'torchvision>=0.2.1'], **args)
