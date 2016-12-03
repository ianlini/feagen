import os
from tempfile import NamedTemporaryFile

import h5py

from .generate_lifetime_features import generate_lifetime_features


def test_generate_lifetime_features():
    with NamedTemporaryFile(suffix=".h5", delete=False) as fp:
        global_feature_hdf_path = fp.name
    with NamedTemporaryFile(suffix=".h5", delete=False) as fp:
        concat_feature_hdf_path = fp.name
    generate_lifetime_features(global_feature_hdf_path, concat_feature_hdf_path)

    with h5py.File(global_feature_hdf_path, "r") as global_feature_h5f, \
            h5py.File(concat_feature_hdf_path, "r") as concat_feature_h5f:
        feature_set = {'weight', 'height', 'BMI', 'weight_divided_by_height'}
        label_set = {'label'}
        test_filter_set = {'is_in_test_set'}
        assert (set(global_feature_h5f)
                == feature_set | label_set | test_filter_set)
        assert (set(concat_feature_h5f)
                == {'feature', 'test_filter_list', 'label'})
        assert set(concat_feature_h5f['label']) == label_set
        assert set(concat_feature_h5f['test_filter_list']) == test_filter_set

    os.remove(global_feature_hdf_path)
    os.remove(concat_feature_hdf_path)
