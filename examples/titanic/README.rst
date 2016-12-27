*******
Titanic
*******

This example uses the `Titanic competition <https://www.kaggle.com/c/titanic>`_
from Kaggle to demonstrate how to utilize *feagen* to organized your feature for
each submission.

Project structure
=================

A project using feagen consist the following:

* hdf file: storing all generated feature
* a feature generator inherited from feagen.FeatureGenerator: this class contains
  the implementation of how each feature is generated
* .feagenrc/config.yml: sets the path to the hdf file, bundles files, and
  the feature generator to use
* multiple .feagenrc/bundle_config.yml: each defines how to bundle a set of
  features

Thinking in feagen
==================


Data Download
=============

Go down to path feagen/examples/titanic and download the dataset (train.csv,
test.csv) from https://www.kaggle.com/c/titanic/data to the directory data.

Getting Started
###############

Create the feagen configuration file template by:

.. code-block:: bash

   feagen-init

This command will create a directory .feagenrc in your current directory. In
.feagenrc/config.yml, change the generator_class key to
titanic.TitanicFeatureGenerator, which would be the class name for
FeatureGenerator.  For the key generator_kwargs, we set the path to the data and
where the data to be stored (like the following). These arguments will be used
to initialize the FeatureGenerator.

.. code-block:: yaml

    generator_kwargs:
      h5py_hdf_path:
        data/global_data.h5
      data_train_csv_path:
        data/train.csv
      data_test_csv_path:
        data/test.csv
    
We will leave .feagenrc/bundle_config.yml for now.

Implement FeatureGenerator
==========================

The idea for FeatureGenerator is that the process of generating each components
for a project can be organized into a
`Directed acyclic graph (DAG) <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`_.
Each methods in FeatureGenerator uses the decorator @require to define which
method to be run first.

We will started by the implementation of FeatureGenerator.

1. In __init__ method, we need to pass in the path to the downloaded data.

.. code-block:: python

    class TitanicFeatureGenerator(fg.FeatureGenerator):
        def __init__(self, h5py_hdf_path, data_train_csv_path, data_test_csv_path):
            super(TitanicFeatureGenerator, self).__init__(h5py_hdf_path)
            self.data_train_csv_path = data_train_csv_path
            self.data_test_csv_path = data_test_csv_path

2. Load data into memory. The first argument in the @will_generate decortor
   declares which data_handler to use. "memory" indicates that the returned data
   will be stored in memory.  The second argument in @will_generate will be used
   as a key for 

.. code-block:: python

    @will_generate('memory', 'data_df')
    def gen_data_df(self):
        trn_df = pd.read_csv(self.data_train_csv_path, index_col='PassengerId')
        tst_df = pd.read_csv(self.data_test_csv_path, index_col='PassengerId')
        tst_df['Survived'] = -1 # additional column to test data
        return {'data_df': pd.concat([trn_df, tst_df])}

   By returning a dictionary, the methods that requires this method will receive
   it from its second argument. For the example bwlow, the second argument for gen_parch is
   exactly what gen_data_df has returned. If requires multiple methods, the
   returned dictionary will be combined into one.

.. code-block:: python

    @require('data_df')
    @will_generate('h5py', 'parch')
    def gen_parch(self, data):

3. The information required fore training includes 'passenger_id' for outputing,
   "label", "is_test", "is_valid" for validation and prediction, and features to be
   trained by the model

.. code-block:: python
    @require('data_df')
    @will_generate('h5py', 'passenger_id')
    def gen_passenger_id(self, data):
        data_df = data['data_df']
        return {'passenger_id': data_df.index.values}

.. code-block:: python

    @require('data_df')
    @will_generate('h5py', 'label')
    def gen_label(self, data):
        data_df = data['data_df']
        return {'label': data_df['Survived'].values}

.. code-block:: python

    @require('data_df')
    @will_generate('h5py', 'is_test')
    def gen_is_test(self, data):
        return {'is_test': (data['data_df']['Survived'].values == -1)}

.. code-block:: python

    @require('data_df')
    @will_generate('h5py', 'is_valid')
    def gen_is_validation(self, data):
        from sklearn.model_selection import train_test_split
        import numpy as np
        data_df = data['data_df']
        df = pd.DataFrame(False, index=data_df.index, columns=['is_valid'], dtype=bool)
        random_state = np.random.RandomState(1126)
        _, valid_id = train_test_split(
            data_df.loc[(data_df['Survived']!=-1)].index, test_size=0.3,
            random_state=random_state)
        df.loc[valid_id, 'is_valid'] = True
        return df

  note that the return value don't have to be a dictionary, it just has to have
  the method keys and __getitem__.  In this case, the default h5py data handler
  will write all the keys in df to the hdf5 dataset.


4. Ways to build features

.. code-block:: python

    @require('data_df')
    @will_generate('h5py', 'parch')
    def gen_parch(self, data):
        data_df = data['data_df']
        return {'parch': data_df['Parch']}

.. code-block:: python

    @require('data_df')
    @will_generate('h5py', 'pclass')
    def gen_pclass(self, data):
        from sklearn.preprocessing import OneHotEncoder
        data_df = data['data_df']
        pclass = np.array(data_df['Pclass'].values)
        pclass[np.isnan(pclass)] = 4 # unknown as a class
        return {'pclass': OneHotEncoder(sparse=False)
                          .fit_transform(pclass.reshape((-1, 1)))}

.. code-block:: python

    @require('data_df')
    @will_generate('h5py', 'family_size')
    def gen_family_size(self, data):
        data_df = data['data_df']
        return {'family_size': (data_df['SibSp'] + data_df['Parch']).values}

You may generate multiple features at a time.

.. code-block:: python

    @require('data_df')
    @will_generate('h5py', ['age', 'sibsp'])
    def gen_age_sibsp(self, data):
        data_df = data['data_df'].copy()
        # clean up age data
        data_df.loc[data_df['Age'].isnull(), 'Age'] = data_df['Age'].mean()
        return {'age': data_df['Age'].values,
                'sibsp': data_df['SibSp'].values}

Bundle
======

Bundle lets the user control which set of feature to use.

.. code-block:: bash

   cp ./.feagenrc/bundle_config.yml ./.feagenrc/feature01.yml

In feature01.yml, we need to define the name of this feature set and which
components generated from FeatureGenerator to include in this bundle. An example 
which includes all the features we have generated previously.

.. code-block:: yaml

    name: feature01
    structure:
      id: passenger_id
      label: label
      info:
      - is_test
      - is_valid
      features:
      - family_size
      - sibsp
      - age
      - pclass
    structure_config:
      features:
        concat: True

The benifit of using bundle is that it lets users to reuse previously generated
features, controls which set of feature has been experimented before and lets
multiple users able to cooperate with each other.

Feature Generation (from command line)
======================================

After all the configuration, run the following command will start the feature
generation process.

.. code-block:: bash

    feagen -b .feagenrc/feature01.yml 

All the feature generated will be stored in data/global_feature.h5 and the file
which bundles feature01 appears in data_bundles/feature01.h5.

The generated bundle hdf5 will have the following structure:

.. code-block:: ipython

	In [1]: import h5py

	In [2]: f = h5py.File('./data_bundles/feature01.h5', 'r')

	In [3]: f.visit(print)
	features
	id
	info
	info/is_test
	info/is_valid
	label

Feature Generation (from Python)
================================

Define the features to be generated and declare the FeatureGenerator.

.. code-block:: python

    generator = TitanicFeatureGenerator(h5py_hdf_path,
        os.path.join(os.path.abspath(__file__), 'data', 'train.csv'),
        os.path.join(os.path.abspath(__file__), 'data', 'test.csv'))
    label_list = ['label']
    info_list = ['is_valid', 'is_test']
    id_list = ['passenger_id']
    feature_list = ['family_size', 'sibsp', 'age', 'pclass']

Generate the feature by the generate method and pass in the name of the features
to be generated.

.. code-block:: python

    generator.generate(feature_list + label_list + info_list)

Define the structure of the bundle and their configuration. Generate the bundle
by the bundle method. (The following structure will the same as the setting in
.feagenrc/feature01.yml)

.. code-block:: python

    bundle_structure = {'label': label_list[0],
                        'info': info_list,
                        'id': id_list[0],
                        'features': feature_list}
    structure_config = {'features': {'concat': True}}
    generator.bundle(bundle_structure, data_bundle_hdf_path=bundle_hdf_path,
            structure_config=structure_config)

Train Model
===========

Load the bundled data.

.. code-block:: python

    bundle_hdf_path = os.path.join(
        os.path.dirname(__file__), 'data_bundles', 'feature01.h5')
    bundle_f = h5py.File(bundle_hdf_path, 'r')

Retrieve the data.

.. code-block:: python

    with h5py.File(bundle_hdf_path, 'r') as bundle_f:
        is_valid = bundle_f['info']['is_valid'].value
        is_test = bundle_f['info']['is_test'].value
        passenger_id = bundle_f['id'].value
        label = bundle_f['label'].value

The feature with structure config concat True is loaded this way.

.. code-block:: python

        feature = np.array(bundle_f['features'])

Set the filter for data.

.. code-block:: python

    train_filter = (np.bitwise_and(~is_valid, ~is_test))
    valid_filter = (np.bitwise_and(is_valid, ~is_test))
    test_filter = is_test

Evaluate the validation set.

.. code-block:: python

    clf = RandomForestClassifier()
    clf.fit(feature[train_filter], label[train_filter])
    print('validation score: (Accuracy)',
          clf.score(feature[valid_filter], label[valid_filter]))

Output the prediction to file and ready to submit to kaggle.

.. code-block:: python

    prediction = clf.predict(feature[test_filter])

    df = pd.DataFrame(prediction, columns=['Survived'],
                      index=passenger_id[test_filter])
    df.index.rename('PassengerId')
    df.to_csv(prediction_csv_path)
