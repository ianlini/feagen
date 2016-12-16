*******
Titanic
*******

This example uses the `Titanic competition <https://www.kaggle.com/c/titanic>`_
from Kaggle to demonstrate how to utilize *feagen* to organized your feature for
each submission.

Data Download
=============

Go down to path feagen/examples/titanic and download the dataset (train.csv,
test.csv) from https://www.kaggle.com/c/titanic/data to the directory data.


Configuration File
##################

.. code-block:: bash
   feagen-init




2. Run gen_feature.py and it will generate two files data/data.h5
   data/concat_feature.h5.  data.h5 contains all features that are extracted
   from the dataset and concat_feature lets you explore through different
   combinations of features to be fed into the model.

3. Run run_model.py and it will extract information from data/concat_feature.h5
   and train a model from it.
