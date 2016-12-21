from past.builtins import basestring

import numpy as np
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

    def fill_concat_data(self, dset_name, data_keys, bundle_h5_group,
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

        bundle_h5_group.create_dataset(dset_name, shape=concat_shape,
                                       dtype=np.float32)

        data_d = 0
        dset = bundle_h5_group[dset_name]
        for data_i, (data_key, data_shape) in enumerate(
                zip(data_keys, data_shapes)):
            batch_size = (buffer_size
                          // (self.get(data_key).dtype.itemsize
                              * data_shape[1]))
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
                    data_buffer = self.get(data_key)[batch_start: batch_end]
                    if len(data_buffer.shape) == 1:
                        data_buffer = data_buffer[:, np.newaxis]
                    dset[batch_start: batch_end,
                         data_d: data_d + data_shape[1]] = data_buffer

            data_d += data_shape[1]

    def bundle(self, structure, data_bundle_hdf_path, buffer_size=int(1e+9),
               structure_config=None):
        if structure_config is None:
            structure_config = {}

        def _bundle_data(structure, structure_config, bundle_h5_group):
            for key, val in six.viewitems(structure):
                config = structure_config.get(key, {})
                if isinstance(val, basestring):
                    bundle_h5_group.create_dataset(key, data=self.get(val))
                elif isinstance(val, list):
                    if config.get('concat', False):
                        self.fill_concat_data(key, val, bundle_h5_group,
                                              buffer_size)
                    else:
                        new_group = bundle_h5_group.create_group(key)
                        for data_key in val:
                            new_group.create_dataset(data_key,
                                                     data=self.get(data_key))
                elif isinstance(val, dict):
                    new_group = bundle_h5_group.create_group(key)
                    _bundle_data(structure[key], structure_config.get(key, {}),
                                 new_group)
                else:
                    raise TypeError("The bundle structure only support "
                                    "dict, list and str.")

        with h5py.File(data_bundle_hdf_path, 'w') as data_bundle_h5f, \
                SimpleTimer("Bundling data"):
            _bundle_data(structure, structure_config, data_bundle_h5f)
