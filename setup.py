#!/usr/bin/env python
import os
from setuptools import setup


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
        'six',
        'future',
        'mkdir-p',
        'h5py',
        'bistiming>=0.1.1',
        'numpy',
        'networkx',
        'pyyaml',
    ]
    tests_require = []


description = """\
A fast and memory-efficient Python feature generating framework for \
machine learning."""

long_description = """\
Please visit  the `GitHub repository <https://github.com/ianlini/feagen>`_
for more information.\n
"""
with open('README.rst') as fp:
    long_description += fp.read()


setup(
    name='feagen',
    version="1.0.0a2",
    description=description,
    long_description=long_description,
    author='Ian Lin',
    url='https://github.com/ianlini/feagen',
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=tests_require,
    license="BSD 2-Clause License",
    entry_points={
        'console_scripts': [
            'feagen = feagen.tools:feagen_run',
            'feagen-init = feagen.tools:init_config',
            'feagen-draw-dag = feagen.tools:draw_full_dag',
        ],
    },
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'License :: OSI Approved :: BSD License',
    ],
    test_suite='nose.collector',
    packages=[
        'feagen',
        'feagen.tools',
    ],
)
