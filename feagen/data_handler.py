import os.path

import six
import h5py
from mkdir_p import mkdir_p


class DataHandler(object):

    @property
    def data_type(self):
        return self.__class__.__name__


class HDF5DataHandler(DataHandler):

    def __init__(self, hdf_path):
        hdf_dir = os.path.dirname(hdf_path)
        if hdf_dir != '':
            mkdir_p(hdf_dir)
        self.h5f = h5py.File(hdf_path, 'a')

    @property
    def data_type(self):
        return 'hdf5_data'

    def can_skip(self, new_data_keys):
        new_data_key_set = set(map(lambda s: "/" + s, new_data_keys))
        generated_set = set(self.h5f.keys())
        if new_data_key_set <= generated_set:
            return True
        return False

    def get(self, keys):
        if isinstance(keys, str):
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


class MemoryIntermediateDataHandler(DataHandler):

    def __init__(self):
        self.data = {}

    @property
    def data_type(self):
        return 'memory_intermediate_data'

    def can_skip(self, new_data_keys):
        new_data_key_set = set(new_data_keys)
        generated_set = set(six.viewkeys(self.data))
        if new_data_key_set <= generated_set:
            return True
        return False

    def get(self, keys):
        if isinstance(keys, str):
            keys = (keys,)
        return {key: self.data[key] for key in keys}

    def get_function_kwargs(self, will_generate_keys, data):  # pylint: disable=unused-argument
        kwargs = {}
        if len(data) > 0:
            kwargs['data'] = data
        return kwargs
