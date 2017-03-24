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

from .pandas_hdf import PandasHDFDataset


SPARSE_FORMAT_SET = set(['csr', 'csc'])


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

    def bundle(self, key, path, new_key):
        """Copy the data to another HDF5 file with new key."""
        data = self.get(key)
        with h5py.File(path) as h5f:
            if ss.isspmatrix(data) or isinstance(data, h5sparse.Dataset):
                h5f = h5sparse.Group(h5f)
            h5f.create_dataset(new_key, data=data)


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

    def get(self, key):
        if isinstance(key, basestring):
            return h5sparse.Group(self.h5f)[key]
        return {k: h5sparse.Group(self.h5f)[k] for k in key}

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
        elif manually_create_dataset in SPARSE_FORMAT_SET:
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
        for key, result in six.iteritems(result_dict):
            if ss.isspmatrix(result):
                if np.isnan(result.data).any():
                    raise ValueError("data {} have nan".format(key))
            elif np.isnan(result).any():
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

    def get(self, key):
        if isinstance(key, basestring):
            return PandasHDFDataset(self.hdf_store, key)
        return {k: PandasHDFDataset(self.hdf_store, k) for k in key}

    def get_function_kwargs(self, will_generate_keys, data,
                            manually_append=False):
        kwargs = {}
        if len(data) > 0:
            kwargs['data'] = data
        if manually_append is True:
            kwargs['append_functions'] = {
                k: partial(self.hdf_store.append, k)
                for k in will_generate_keys
            }
        return kwargs

    def check_result_dict_keys(self, result_dict, will_generate_keys,
                               function_name, handler_key,
                               manually_append=False):
        will_generate_key_set = set(will_generate_keys)
        result_dict_key_set = set(result_dict.keys())
        if manually_append:
            check_redundant_keys(result_dict_key_set, will_generate_key_set,
                                 function_name, handler_key)
            # TODO: check all the datasets is either manually created or in
            #       result_dict_key_set
        else:
            check_exact_match_keys(result_dict_key_set, will_generate_key_set,
                                   function_name, handler_key)

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
                if (isinstance(result, pd.DataFrame)
                        and isinstance(result.index, pd.MultiIndex)
                        and isinstance(result.columns, pd.MultiIndex)):
                    self.hdf_store.put(key, result)
                else:
                    self.hdf_store.put(key, result, format='table')
        self.hdf_store.flush(fsync=True)

    def bundle(self, key, path, new_key):
        """Copy the data to another HDF5 file with new key."""
        data = self.get(key).value
        data.to_hdf(path, new_key)


class MemoryDataHandler(DataHandler):

    def __init__(self):
        self.data = {}

    def can_skip(self, data_key):
        if data_key in self.data:
            return True
        return False

    def get(self, key):
        if isinstance(key, basestring):
            return self.data[key]
        return {k: self.data[k] for k in key}

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

    def get(self, key):
        if isinstance(key, basestring):
            with open(os.path.join(self.pickle_dir, key + ".pkl"), "rb") as fp:
                return cPickle.load(fp)
        data = {}
        for k in key:
            with open(os.path.join(self.pickle_dir, k + ".pkl"), "rb") as fp:
                data[k] = cPickle.load(fp)
        return data

    def write_data(self, result_dict):
        for key, val in six.viewitems(result_dict):
            pickle_path = os.path.join(self.pickle_dir, key + ".pkl")
            with SimpleTimer("Writing generated data %s to pickle file" % key,
                             end_in_new_line=False), \
                    open(pickle_path, "wb") as fp:
                cPickle.dump(val, fp, protocol=cPickle.HIGHEST_PROTOCOL)
