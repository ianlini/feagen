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
        for _, val in six.viewitems(structure):
            if isinstance(val, basestring):
                data_keys.append(val)
            elif isinstance(val, list):
                data_keys.extend(val)
            elif isinstance(val, dict):
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
            data_shape = self.get(data_key).shape
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

        def _bundle_data(structure, structure_config, group_name):
            for key, val in six.viewitems(structure):
                config = structure_config.get(key, {})
                if isinstance(val, basestring):
                    (self.get_handler(val)
                     .bundle(val, data_bundle_hdf_path, group_name + "/" + key))
                elif isinstance(val, list):
                    if config.get('concat', False):
                        self.fill_concat_data(
                            data_bundle_hdf_path, group_name + "/" + key,
                            val, buffer_size)
                    else:
                        for data_key in val:
                            (self.get_handler(data_key)
                             .bundle(data_key, data_bundle_hdf_path,
                                     "%s/%s/%s" % (group_name, key, data_key)))
                elif isinstance(val, dict):
                    _bundle_data(structure[key], structure_config.get(key, {}),
                                 group_name + "/" + key)
                else:
                    raise TypeError("The bundle structure only support "
                                    "dict, list and str.")

        if os.path.isfile(data_bundle_hdf_path):
            os.remove(data_bundle_hdf_path)
        with SimpleTimer("Bundling data"):
            _bundle_data(structure, structure_config, "/")
