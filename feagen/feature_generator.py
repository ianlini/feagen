import os.path

import h5py
from mkdir_p import mkdir_p
import six


class FeatureGeneratorType(type):

    def __new__(cls, clsname, bases, dct):
        import ipdb; ipdb.set_trace()
        # if func.__name__ in func_dict.values():
        #     raise ValueError("duplicated function name " + func.__name__)
        # for key in will_generate_keys:
        #     if key in func_dict:
        #         raise ValueError("duplicated {} {} in {} and {}".format(
        #             set_name, key, func_dict[key], func.__name__))
        #     func_dict[key] = func.__name__
        return super(FeatureGeneratorType, cls).__new__(
            cls, clsname, bases, dct)


class FeatureGenerator(six.with_metaclass(FeatureGeneratorType, object)):

    def __init__(self, global_feature_hdf_path, feature_func_dict):
        self.intermediate_data = {}
        global_feature_hdf_dir = os.path.dirname(global_feature_hdf_path)
        if global_feature_hdf_dir != '':
            mkdir_p(global_feature_hdf_dir)
        self.global_feature_h5f = h5py.File(global_feature_hdf_path, 'a')
        self.feature_func_dict = feature_func_dict

    def generate(self, feature_names):
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        self.require(feature_names, self.feature_func_dict)

    def require(self, keys, func_dict):
        for key in keys:
            getattr(self, func_dict[key])(key)
