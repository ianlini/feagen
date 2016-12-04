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

    def can_skip(self, new_data_key_set):
        new_data_key_set = set(map(lambda s: "/" + s, new_data_key_set))
        generated_set = set(self.h5f.keys())
        if new_data_key_set <= generated_set:
            return True
        return False

    def get(self, keys):
        if isinstance(keys, str):
            keys = (keys,)
        return {key: self.h5f[key] for key in keys}

class MemoryIntermediateDataHandler(DataHandler):
    def __init__(self):
        self.data = {}

    @property
    def data_type(self):
        return 'memory_intermediate_data'

    def can_skip(self, new_data_key_set):
        generated_set = set(six.viewkeys(self.data))
        if new_data_key_set <= generated_set:
            return True
        return False

    def get(self, keys):
        if isinstance(keys, str):
            keys = (keys,)
        return {key: self.data[key] for key in keys}
