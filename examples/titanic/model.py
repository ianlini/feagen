
import os

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def load_feature_run_model(bundle_hdf_path, prediction_csv_path):
    with h5py.File(bundle_hdf_path, 'r') as bundle_f:
        is_valid = bundle_f['info']['is_valid'].value
        is_test = bundle_f['info']['is_test'].value
        passenger_id = bundle_f['id'].value
        label = bundle_f['label'].value

        # concated
        feature = bundle_f['features'].value

    train_filter = (np.bitwise_and(~is_valid, ~is_test))
    valid_filter = (np.bitwise_and(is_valid, ~is_test))
    test_filter = is_test

    ##############
    # validation #
    ##############
    clf = RandomForestClassifier()
    clf.fit(feature[train_filter], label[train_filter])
    print('validation score: (Accuracy)',
          clf.score(feature[valid_filter], label[valid_filter]))

    ##############
    # prediction #
    ##############
    clf.fit(feature[~test_filter], label[~test_filter])
    prediction = clf.predict(feature[test_filter])

    df = pd.DataFrame(prediction, columns=['Survived'],
                      index=passenger_id[test_filter])
    df.index.rename('PassengerId')
    df.to_csv(prediction_csv_path)


if __name__ == '__main__':
    bundle_hdf_path = os.path.join(
        os.path.dirname(__file__), 'data_bundles', 'feature01.h5')
    prediction_csv_path = 'prediction.csv'
    load_feature_run_model(bundle_hdf_path, prediction_csv_path)
