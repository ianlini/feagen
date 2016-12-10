Feagen
======
.. image:: https://img.shields.io/travis/ianlini/feagen/master.svg
   :target: https://travis-ci.org/ianlini/feagen
.. image:: https://img.shields.io/pypi/v/feagen.svg
   :target: https://pypi.python.org/pypi/feagen
.. image:: https://img.shields.io/pypi/l/feagen.svg
   :target: https://pypi.python.org/pypi/feagen

A fast and memory-efficient Python feature generating framework for machine learning.

Introduction
------------
TODO

Installation
------------
.. code:: bash

   pip install feagen

Getting start
-------------
Getting start from the simple `lifetime prediction example </examples/lifetime_prediction>`_ is an easy way. You can first look at the raw data `lifetime.csv </examples/lifetime_prediction/lifetime.csv>`_ and then you may better understand what we are doing to the data.

Creating the feature generator
******************************
The most important part in Feagen is the feature generator class.
You first need to define a class like in `lifetime_feature_generator.py </examples/lifetime_prediction/lifetime_feature_generator.py>`_ to tell Feagen how to deal with the data.
(TODO: more details)

Creating the config files
*************************
There is a command line tool ``feagen-init`` that can help create the initial config files: the global config ``.feagenrc/config.yml`` and the bundle config ``.feagenrc/bundle_config.yml``.
You can look at the comments that are automatically generated in those files or in `examples/lifetime_prediction/.feagenrc </examples/lifetime_prediction/.feagenrc>`_ to understand how to change them.
(TODO: more details)

Drawing the directed acyclic graph (DAG)
****************************************
There is one way for you to check if the dependency is correct.
You can use the command line tool ``feagen-draw-dag`` to draw the DAG image:

   usage: feagen-draw-dag [-h] [-g GLOBAL_CONFIG] [-d DAG_OUTPUT_PATH]

   Generate DAG.

   optional arguments:
     -h, --help            show this help message and exit
     -g GLOBAL_CONFIG, --global-config GLOBAL_CONFIG
                           the path of the path configuration YAML file (default:
                           .feagenrc/config.yml)
     -d DAG_OUTPUT_PATH, --dag-output-path DAG_OUTPUT_PATH
                           output image path (default: dag.png)

You can specify the paths of the global config and the output image using ``-g`` and ``-d`` respectively.
Running ``feagen-draw-dag`` in ``examples/lifetime_prediction`` will give you `examples/lifetime_prediction/dag.png </examples/lifetime_prediction/dag.png>`_:

.. image:: /examples/lifetime_prediction/dag.png

(Note that the order may not be the same)
