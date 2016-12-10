import gc

import numpy as np
import h5py
import six
from bistiming import IterTimer, SimpleTimer


def flatten_structure(structure):
    structure = structure.copy()
    data_keys = structure.pop('features')[:]
    def _flatten_structure(structure, data_keys):
        for _, val in six.viewitems(structure):
            if isinstance(val, str):
                data_keys.append(val)
            elif isinstance(val, list):
                data_keys.extend(val)
            elif isinstance(val, dict):
                _flatten_structure(val, data_keys)
            else:
                raise TypeError("The bundle structure only support "
                                "dict, list and str.")
        return data_keys
    return _flatten_structure(structure, data_keys)


def fill_concat_features(feature_list, global_data_h5f, data_bundle_h5f,
                         buffer_size=int(1e+9)):
    feature_shapes = []
    for feature_name in feature_list:
        feature_shape = global_data_h5f[feature_name].shape
        if len(feature_shape) == 1:
            feature_shape += (1,)
        feature_shapes.append(feature_shape)
    max_shape_length = max(map(len, feature_shapes))
    if max_shape_length > 2:
        raise NotImplementedError("tensor feature is not supported yet")

    n_instances = feature_shapes[0][0]
    for feature_shape in feature_shapes:
        if feature_shape[0] != n_instances:
            raise ValueError("different number of instances: {} and {}."
                             .format(feature_shapes[0], feature_shape))
    n_features = sum(shape[1] for shape in feature_shapes)
    concat_shape = (n_instances, n_features)

    data_bundle_h5f.create_dataset('features', shape=concat_shape,
                                   dtype=np.float32)

    feature_d = 0
    dset = data_bundle_h5f['features']
    for feature_i, (feature_name, feature_shape) in enumerate(
            zip(feature_list, feature_shapes)):
        batch_size = (buffer_size
                      // (global_data_h5f[feature_name].dtype.itemsize
                          * feature_shape[1]))
        if batch_size == 0:
            print("Warning! buffer_size not enough to fitted by an "
                  "instance. Trying to use more memory.")
            batch_size = 1
        with IterTimer("({}/{}) Filling {}"
                       .format(feature_i + 1, len(feature_list), feature_name),
                       feature_shape[0]) as timer:
            for batch_start in range(0, feature_shape[0], batch_size):
                timer.update(batch_start)
                batch_end = min(feature_shape[0], batch_start + batch_size)
                feature_buffer = (global_data_h5f[feature_name]
                                  [batch_start: batch_end])
                if len(feature_buffer.shape) == 1:
                    feature_buffer = feature_buffer[:, np.newaxis]
                dset[batch_start: batch_end,
                     feature_d: feature_d + feature_shape[1]] = feature_buffer

        feature_d += feature_shape[1]


def bundle_data(structure, global_data_hdf_path, data_bundle_hdf_path,
                buffer_size=int(1e+9)):
    structure = structure.copy()
    feature_list = structure.pop('features')

    def _bundle_data(structure, group, global_data_h5f):
        for key, val in six.viewitems(structure):
            if isinstance(val, str):
                group.create_dataset(key, data=global_data_h5f[val])
            elif isinstance(val, list):
                new_group = group.create_group(key)
                for data_key in val:
                    new_group.create_dataset(data_key,
                                             data=global_data_h5f[data_key])
            elif isinstance(val, dict):
                new_group = group.create_group(key)
                _bundle_data(structure[key], new_group, global_data_h5f)
            else:
                raise TypeError("The bundle structure only support "
                                "dict, list and str.")

    with h5py.File(global_data_hdf_path, 'r') as global_data_h5f, \
            h5py.File(data_bundle_hdf_path, 'w') as data_bundle_h5f, \
            SimpleTimer("Concatenating and saving features"):
        _bundle_data(structure, data_bundle_h5f, global_data_h5f)
        # save extra features
        fill_concat_features(feature_list, global_data_h5f,
                             data_bundle_h5f, buffer_size)

    gc.collect()
