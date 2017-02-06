import os.path
from abc import ABCMeta, abstractmethod
from functools import partial

from bistiming import SimpleTimer
import h5py
import h5sparse
from mkdir_p import mkdir_p
import numpy as np
import pandas as pd
from past.builtins import basestring
import scipy.sparse as ss
import six
from six.moves import cPickle


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


class DataHandler(six.with_metaclass(ABCMeta, object)):

    @abstractmethod
    def can_skip(self, data_key):
        pass

    @abstractmethod
    def get(self, keys):
        pass

    def get_function_kwargs(self, will_generate_keys, data):
        # pylint: disable=unused-argument
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

    @abstractmethod
    def write_data(self, result_dict):
        pass


class H5pyDataHandler(DataHandler):

    def __init__(self, hdf_path):
        hdf_dir = os.path.dirname(hdf_path)
        if hdf_dir != '':
            mkdir_p(hdf_dir)
        self.h5f = h5py.File(hdf_path, 'a')

    def can_skip(self, data_key):
        if data_key in self.h5f:
            return True
        return False

    def get(self, keys):
        if isinstance(keys, basestring):
            keys = (keys,)
        return {key: h5sparse.Group(self.h5f)[key] for key in keys}

    def get_function_kwargs(self, will_generate_keys, data,
                            manually_create_dataset=False):
        kwargs = {}
        if len(data) > 0:
            kwargs['data'] = data
        if manually_create_dataset is True:
            kwargs['create_dataset_functions'] = {
                k: partial(self.h5f.create_dataset, k)
                for k in will_generate_keys
            }
        elif manually_create_dataset == "csr":
            kwargs['create_dataset_functions'] = {
                k: partial(h5sparse.Group(self.h5f).create_dataset, k)
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
        no_nan_check_dtype = [np.bool]
        for key, result in six.iteritems(result_dict):
            if (result.dtype not in no_nan_check_dtype
                    and np.isnan(result).any()):
                raise ValueError("data {} have nan".format(key))
            with SimpleTimer("Writing generated data {} to hdf5 file"
                             .format(key),
                             end_in_new_line=False):
                if key in self.h5f:
                    # self.h5f[key][...] = result
                    raise NotImplementedError("Overwriting not supported.")
                else:
                    if (isinstance(result, ss.csc_matrix)
                            or isinstance(result, ss.csr_matrix)):
                        # sparse matrix
                        h5sparse.Group(self.h5f).create_dataset(key,
                                                                data=result)
                    else:
                        self.h5f.create_dataset(key, data=result)
        self.h5f.flush()


class PandasHDFDataHandler(DataHandler):

    def __init__(self, hdf_path):
        hdf_dir = os.path.dirname(hdf_path)
        if hdf_dir != '':
            mkdir_p(hdf_dir)
        self.hdf_store = pd.HDFStore(hdf_path)

    def can_skip(self, data_key):
        if data_key in self.hdf_store:
            return True
        return False

    def get(self, keys):
        if isinstance(keys, basestring):
            keys = (keys,)
        return {key: self.hdf_store[key] for key in keys}

    def write_data(self, result_dict):
        for key, result in six.iteritems(result_dict):
            is_null = False
            if isinstance(result, pd.DataFrame):
                if result.isnull().any().any():
                    is_null = True
            elif isinstance(result, pd.Series):
                if result.isnull().any():
                    is_null = True
            else:
                raise ValueError("PandasHDFDataHandler doesn't support type "
                                 "{} (in key {})".format(type(result), key))
            if is_null:
                raise ValueError("data {} have nan".format(key))
            with SimpleTimer("Writing generated data {} to hdf5 file"
                             .format(key),
                             end_in_new_line=False):
                self.hdf_store[key] = result
        self.hdf_store.flush(fsync=True)


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

    def write_data(self, result_dict):
        self.data.update(result_dict)


class PickleDataHandler(DataHandler):

    def __init__(self, pickle_dir):
        mkdir_p(pickle_dir)
        self.pickle_dir = pickle_dir

    def can_skip(self, data_key):
        data_path = os.path.join(self.pickle_dir, data_key + ".pkl")
        if os.path.exists(data_path):
            return True
        return False

    def get(self, keys):
        if isinstance(keys, basestring):
            keys = (keys,)
        data = {}
        for key in keys:
            with open(os.path.join(self.pickle_dir, key + ".pkl"), "rb") as fp:
                data[key] = cPickle.load(fp)
        return data

    def write_data(self, result_dict):
        for key, val in six.viewitems(result_dict):
            with open(os.path.join(self.pickle_dir, key + ".pkl"), "wb") as fp:
                cPickle.dump(val, fp, protocol=cPickle.HIGHEST_PROTOCOL)
