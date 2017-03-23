import os
from past.builtins import basestring

import numpy as np
import pandas as pd
import h5py
import six
from bistiming import IterTimer, SimpleTimer


def get_data_keys_from_structure(structure):
    data_keys = []

    def _get_data_keys_from_structure(structure):
        if isinstance(structure, basestring):
            data_keys.append(structure)
        elif isinstance(structure, list):
            data_keys.extend(structure)
        elif isinstance(structure, dict):
            for _, val in six.viewitems(structure):
                _get_data_keys_from_structure(val)
        else:
            raise TypeError("The bundle structure only support "
                            "dict, list and str.")
    _get_data_keys_from_structure(structure)

    return data_keys


class DataBundlerMixin(object):

    def fill_concat_data(self, data_bundle_hdf_path, dset_name, data_keys,
                         buffer_size=int(1e+9)):
        data_shapes = []
        for data_key in data_keys:
            data = self.get(data_key)
            if hasattr(data, '__call__'):
                data = data()
            data_shape = data.shape
            if len(data_shape) == 1:
                data_shape += (1,)
            data_shapes.append(data_shape)
        max_shape_length = max(map(len, data_shapes))
        if max_shape_length > 2:
            raise NotImplementedError("tensor data is not supported yet")

        n_rows = data_shapes[0][0]
        for data_shape in data_shapes:
            if data_shape[0] != n_rows:
                raise ValueError("different number of instances: {} and {}."
                                 .format(data_shapes[0], data_shape))
        n_cols = sum(shape[1] for shape in data_shapes)
        concat_shape = (n_rows, n_cols)

        h5f = h5py.File(data_bundle_hdf_path)
        dset = h5f.create_dataset(dset_name, shape=concat_shape,
                                  dtype=np.float32)

        data_d = 0
        for data_i, (data_key, data_shape) in enumerate(
                zip(data_keys, data_shapes)):
            data = self.get(data_key)
            if hasattr(data, '__call__'):
                data = data()
            if isinstance(data, pd.DataFrame):
                data = data.values
            batch_size = buffer_size // (data.dtype.itemsize * data_shape[1])
            if batch_size == 0:
                print("Warning! buffer_size not enough to fitted by an "
                      "instance. Trying to use more memory.")
                batch_size = 1
            with IterTimer("({}/{}) Filling {}"
                           .format(data_i + 1, len(data_keys), data_key),
                           data_shape[0]) as timer:
                for batch_start in range(0, data_shape[0], batch_size):
                    timer.update(batch_start)
                    batch_end = min(data_shape[0], batch_start + batch_size)
                    data_buffer = data[batch_start: batch_end]
                    if len(data_buffer.shape) == 1:
                        data_buffer = data_buffer[:, np.newaxis]
                    dset[batch_start: batch_end,
                         data_d: data_d + data_shape[1]] = data_buffer

            data_d += data_shape[1]

        h5f.close()

    def bundle(self, structure, data_bundle_hdf_path, buffer_size=int(1e+9),
               structure_config=None):
        if structure_config is None:
            structure_config = {}

        def _bundle_data(structure, structure_config, dset_name=""):
            if isinstance(structure, basestring) and dset_name != "":
                (self.get_handler(structure)
                 .bundle(structure, data_bundle_hdf_path, dset_name))
            elif isinstance(structure, list):
                if structure_config.get('concat', False):
                    self.fill_concat_data(data_bundle_hdf_path, dset_name,
                                          structure, buffer_size)
                else:
                    for data_key in structure:
                        (self.get_handler(data_key)
                         .bundle(data_key, data_bundle_hdf_path,
                                 dset_name + "/" + data_key))
            elif isinstance(structure, dict):
                for key, val in six.viewitems(structure):
                    _bundle_data(val, structure_config.get(key, {}),
                                 dset_name + "/" + key)
            else:
                raise TypeError("The bundle structure only support "
                                "dict, list and str (except the first layer).")

        if os.path.isfile(data_bundle_hdf_path):
            os.remove(data_bundle_hdf_path)
        with SimpleTimer("Bundling data"):
            _bundle_data(structure, structure_config)
