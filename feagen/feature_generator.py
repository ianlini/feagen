import os.path

import h5py
from mkdir_p import mkdir_p


class FeatureGenerator(object):

    def __init__(self, global_feature_hdf_path, feature_func_dict):
        self.intermediate_data = {}
        mkdir_p(os.path.dirname(global_feature_hdf_path))
        self.global_feature_h5f = h5py.File(global_feature_hdf_path, 'a')
        self.feature_func_dict = feature_func_dict

    def generate(self, feature_names):
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        self.require(feature_names, self.feature_func_dict)

    def require(self, keys, func_dict):
        for key in keys:
            getattr(self, func_dict[key])(key)
