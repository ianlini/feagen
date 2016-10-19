#!/usr/bin/env python
import os
from setuptools import setup, find_packages


on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
# read the docs could not compile numpy and c extensions
if on_rtd:
    setup_requires = []
    install_requires = []
    tests_require = []
else:
    setup_requires = [
        'nose',
        'coverage',
    ]
    install_requires = [
        'mkdir-p',
        'h5py',
        'bistiming',
        'numpy',
    ]
    tests_require = []

description = ("A fast and memory-efficient Python feature generating "
               "framework for machine learning.")
long_description = ("See `github <https://github.com/ianlini/feagen>`_ "
                    "for more information.")

setup(
    name='feagen',
    version="0.1.0",
    description=description,
    long_description=long_description,
    author='ianlini',
    url='https://github.com/ianlini/feagen',
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=tests_require,
    classifiers=[
        'Topic :: Utilities',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    test_suite='nose.collector',
    packages=find_packages(),
)
