import os.path
from functools import partial
from past.builtins import basestring

import six
import h5py
from mkdir_p import mkdir_p
import numpy as np
from bistiming import SimpleTimer


class DataHandler(object):
    pass


def check_redundant_keys(result_dict_key_set, will_generate_key_set,
                         function_name, handler_key):
    redundant_key_set = result_dict_key_set - will_generate_key_set
    if len(redundant_key_set) > 0:
        raise ValueError("The return keys of function {} {} have "
                         "redundant keys {} while generating {} {}.".format(
                             function_name, result_dict_key_set,
                             redundant_key_set, handler_key,
                             will_generate_key_set))


def check_exact_match_keys(result_dict_key_set, will_generate_key_set,
                           function_name, handler_key):
    if will_generate_key_set != result_dict_key_set:
        raise ValueError("The return keys of function {} {} doesn't "
                         "match {} while generating {}.".format(
                             function_name, result_dict_key_set,
                             will_generate_key_set, handler_key))


class H5pyDataHandler(DataHandler):

    def __init__(self, hdf_path):
        hdf_dir = os.path.dirname(hdf_path)
        if hdf_dir != '':
            mkdir_p(hdf_dir)
        self.h5f = h5py.File(hdf_path, 'a')

    def can_skip(self, data_key):
        if "/" + data_key in self.h5f.keys():
            return True
        return False

    def get(self, keys):
        if isinstance(keys, basestring):
            keys = (keys,)
        return {key: self.h5f[key] for key in keys}

    def get_function_kwargs(self, will_generate_keys, data,
                            manually_create_dataset=False):
        kwargs = {}
        if len(data) > 0:
            kwargs['data'] = data
        if manually_create_dataset:
            kwargs['create_dataset_functions'] = {
                k: partial(self.h5f.create_dataset, k)
                for k in will_generate_keys
            }
        return kwargs

    def check_result_dict_keys(self, result_dict, will_generate_keys,
                               function_name, handler_key,
                               manually_create_dataset=False):
        will_generate_key_set = set(will_generate_keys)
        result_dict_key_set = set(result_dict.keys())
        if manually_create_dataset:
            check_redundant_keys(result_dict_key_set, will_generate_key_set,
                                 function_name, handler_key)
            # TODO: check all the datasets is either manually created or in
            #       result_dict_key_set
        else:
            check_exact_match_keys(result_dict_key_set, will_generate_key_set,
                                   function_name, handler_key)

    def write_data(self, result_dict):
        for key, result in six.iteritems(result_dict):
            if np.isnan(result).any():
                raise ValueError("features {} have nan".format(key))
            with SimpleTimer("Writing generated features {} to hdf5 file"
                             .format(key),
                             end_in_new_line=False):
                if key in self.h5f:
                    self.h5f[key][...] = result
                else:
                    self.h5f.create_dataset(key, data=result)
        self.h5f.flush()


class MemoryDataHandler(DataHandler):

    def __init__(self):
        self.data = {}

    def can_skip(self, data_key):
        if data_key in self.data:
            return True
        return False

    def get(self, keys):
        if isinstance(keys, basestring):
            keys = (keys,)
        return {key: self.data[key] for key in keys}

    def get_function_kwargs(self, will_generate_keys, data):  # pylint: disable=unused-argument
        kwargs = {}
        if len(data) > 0:
            kwargs['data'] = data
        return kwargs

    def check_result_dict_keys(self, result_dict, will_generate_keys,
                               function_name, handler_key):
        will_generate_key_set = set(will_generate_keys)
        result_dict_key_set = set(result_dict.keys())
        check_exact_match_keys(result_dict_key_set, will_generate_key_set,
                               function_name, handler_key)

    def write_data(self, result_dict):
        self.data.update(result_dict)
