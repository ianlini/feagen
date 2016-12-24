
import os

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def load_feature_run_model(bundle_hdf_path, prediction_csv_path):
    bundle_f = h5py.File(bundle_hdf_path, 'r')

    is_valid = np.array(bundle_f['info']['is_valid'])
    is_test = np.array(bundle_f['info']['is_test'])
    passenger_id = np.array(bundle_f['id']['passenger_id'])
    label = np.array(bundle_f['label']['label'])

    # concated
    feature = np.array(bundle_f['features'])

    bundle_f.close()

    train_filter = (np.bitwise_and(is_valid == 0, is_test == 0))
    valid_filter = (np.bitwise_and(is_valid == 1, is_test == 0))
    test_filter = (is_test == 1)

    ##############
    # validation #
    ##############
    clf = RandomForestClassifier()
    clf.fit(feature[train_filter], label[train_filter])
    print('validation score:',
          clf.score(feature[valid_filter], label[valid_filter]))

    ##############
    # prediction #
    ##############
    prediction = clf.predict(feature[test_filter])

    df = pd.DataFrame(prediction, columns=['Survived'],
                      index=passenger_id[test_filter])
    df.index.rename('PassengerId')
    df.to_csv(prediction_csv_path)


if __name__ == '__main__':
    load_feature_run_model(
        os.path.join(os.path.dirname(__file__), 'data_bundles', 'feature01.h5'),
        "prediction.csv")
