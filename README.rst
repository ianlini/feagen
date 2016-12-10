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
Getting start from the simple `lifetime prediction example </examples/lifetime_prediction/>`_ is an easy way.
You can first look at the raw data `lifetime.csv </examples/lifetime_prediction/lifetime.csv>`_ and then you may better understand what we are doing to the data.

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
Running ``feagen-draw-dag -d fig/dag.png`` in `examples/lifetime_prediction/ </examples/lifetime_prediction/>`_ will give you `examples/lifetime_prediction/fig/dag.png </examples/lifetime_prediction/fig/dag.png>`_:

.. image:: /examples/lifetime_prediction/fig/dag.png

(Note that the order may not be the same)

Generating features
*******************
After the generator class and the config are defined, we can now generate the features.
A command line tool ``feagen`` can be used now:

   usage: feagen [-h] [-g GLOBAL_CONFIG] [-b BUNDLE_CONFIG] [-d DAG_OUTPUT_PATH]

   Generate global data and data bundle.

   optional arguments:
     -h, --help            show this help message and exit
     -g GLOBAL_CONFIG, --global-config GLOBAL_CONFIG
                           the path of the path configuration YAML file (default:
                           .feagenrc/config.yml)
     -b BUNDLE_CONFIG, --bundle-config BUNDLE_CONFIG
                           the path of the bundle configuration YAML file
                           (default: .feagenrc/bundle_config.yml)
     -d DAG_OUTPUT_PATH, --dag-output-path DAG_OUTPUT_PATH
                           draw the involved subDAG to the provided path
                           (default: None)

You can specify the paths of the global config, the bundle config, and the involved subDAG image using ``-g``, ``-b`` and ``-d`` respectively.

The program will first find the nodes in the DAG that are involved and build a subDAG for this task, and check whether the data has been generated in the global data.
The resulting DAG after these checks will be output if you specify ``-d``.
For example, in `examples/lifetime_prediction/`_, if you run ``feagen`` first and then add a new feature ``height_divided_by_weight``, and run ``feagen -d fig/involved_dag.png``, you will get an image `examples/lifetime_prediction/fig/involved_dag.png </examples/lifetime_prediction/fig/involved_dag.png>`_:

.. image:: /examples/lifetime_prediction/fig/involved_dag.png

(Note that the order may not be the same)
