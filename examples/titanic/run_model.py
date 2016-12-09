
import os

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def main():
    data_csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    concat_feature_hdf_path = os.path.join(data_csv_dir, 'concat_feature.h5')
    concat_f = h5py.File(concat_feature_hdf_path, 'r')
    global_feature_hdf_path = os.path.join(data_csv_dir, 'data.h5')
    global_f = h5py.File(global_feature_hdf_path, 'r')
    prediction_csv_path = 'prediction.csv'

    is_valid = np.array(concat_f['valid_filter_list']['is_validation'])
    is_test = np.array(concat_f['test_filter_list']['is_test'])
    passenger_id = np.array(concat_f['passenger_id']['passenger_id'])
    feature = np.array(concat_f['feature'])
    label = np.array(concat_f['label']['label'])

    concat_f.close()
    global_f.close()

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
    df.index.name = 'PassengerId'
    df.to_csv(prediction_csv_path)


if __name__ == '__main__':
    main()
