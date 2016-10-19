import gc

import numpy as np
import h5py
from bistiming import IterTimer, SimpleTimer


def fill_concat_features(feature_list, global_feature_h5f, concat_feature_h5f,
                         buffer_size):
    feature_shapes = []
    for feature_name in feature_list:
        feature_shape = global_feature_h5f[feature_name].shape
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

    concat_feature_h5f.create_dataset('feature', shape=concat_shape,
                                      dtype=np.float32)

    feature_d = 0
    dset = concat_feature_h5f['feature']
    for feature_i, (feature_name, feature_shape) in enumerate(
            zip(feature_list, feature_shapes)):
        batch_size = (buffer_size
                      // (global_feature_h5f[feature_name].dtype.itemsize
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
                feature_buffer = (global_feature_h5f[feature_name]
                                  [batch_start: batch_end])
                if len(feature_buffer.shape) == 1:
                    feature_buffer = feature_buffer[:, np.newaxis]
                dset[batch_start: batch_end,
                     feature_d: feature_d + feature_shape[1]] = feature_buffer

        feature_d += feature_shape[1]


def save_concat_features(feature_config, global_feature_hdf_path,
                         buffer_size=int(1e+9)):

    concat_feature_hdf_path = feature_config.concat_feature_hdf_path
    feature_list = feature_config.feature_list

    with h5py.File(global_feature_hdf_path, 'r') as global_feature_h5f, \
            h5py.File(concat_feature_hdf_path, 'w') as concat_feature_h5f, \
            SimpleTimer("Concating and saving features", end_in_new_line=False):
        # save label
        for label in feature_config.label_list:
            concat_feature_h5f.create_dataset(
                'label/' + label, data=global_feature_h5f[label])

        # save extra data
        for data_name in feature_config.extra_data_list:
            concat_feature_h5f.create_dataset(
                'extra/' + data_name, data=global_feature_h5f[data_name])

        # save extra features
        fill_concat_features(feature_list, global_feature_h5f,
                             concat_feature_h5f, buffer_size)

    gc.collect()
