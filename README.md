annom
=======================

This package provides functions associated with anomaly detection, especially related
to pytorch. 

This package extends the functionality of [synthtorch](https://github.com/jcreinhold/synthtorch).
Some of the models/ideas were used to generate the results of the paper [[1](https://arxiv.org/abs/2002.04626)].

Note that this was only tested with python v3.6 and v3.7, so there
may be incompatibility with previous versions.

Requirements
------------

- numpy >= 1.15.4
- torch >= 1.0.0
- torchvision >= 0.2.1

Installation
------------

    python setup.py install

or, if you want to actively develop the package,

    python setup.py develop

Test Package
------------

Unit tests can be run from the main directory as follows:

    nosetests -v tests

Relevant Papers
---------------

[1] J. Reinhold, Y. He, Y. Chen, D. Gao, J. Lee, J. Prince, A. Carass.
    ``Finding novelty with uncertainty.''
    Medical Imaging 2020: Image Processing,
    International Society for Optics and Photonics, 2020.
