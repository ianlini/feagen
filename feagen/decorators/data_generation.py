from __future__ import print_function
from functools import wraps, partial

import six
import numpy as np
from bistiming import SimpleTimer


def require_intermediate_data(data_keys):
    if isinstance(data_keys, str):
        data_keys = (data_keys,)

    def require_intermediate_data_decorator(func):
        @wraps(func)
        def func_wrapper(self, set_name, new_data_key):
            self.require(data_keys, self._intermediate_data_dag)  # pylint: disable=protected-access
            data = {key: self.intermediate_data[key] for key in data_keys}
            func(self, set_name, new_data_key, data=data)
        return func_wrapper
    return require_intermediate_data_decorator


def update_create_dataset_functions(global_feature_h5f, will_generate_keys,
                                    kwargs):
    kwargs['create_dataset_functions'] = {
        k: partial(global_feature_h5f.create_dataset, k)
        for k in will_generate_keys
    }


def check_result_dict_type(result_dict, func_name):
    if not (hasattr(result_dict, 'keys')
            and hasattr(result_dict, '__getitem__')):
        raise ValueError("the return value of mehod {} should have "
                         "keys and __getitem__ methods".format(func_name))


def check_redundant_keys(result_dict_key_set, will_generate_key_set,
                         func_name, set_name, new_data_key):
    redundant_key_set = result_dict_key_set - will_generate_key_set
    if len(redundant_key_set) > 0:
        raise ValueError("the return keys of method {} {} have "
                         "redundant keys {} while generating {} {}".format(
                             func_name, result_dict_key_set, redundant_key_set,
                             set_name, new_data_key))


def check_exact_match_keys(result_dict_key_set, will_generate_key_set,
                           func_name, set_name, new_data_key):
    if will_generate_key_set != result_dict_key_set:
        raise ValueError("the return keys of method {} {} doesn't "
                         "match {} while generating {} {}".format(
                             func_name, result_dict_key_set,
                             will_generate_key_set, set_name,
                             new_data_key))


def check_result_dict_keys(result_dict, will_generate_key_set,
                           func_name, set_name, new_data_key,
                           manually_create_dataset):

    result_dict_key_set = set(result_dict.keys())
    if manually_create_dataset:
        check_redundant_keys(result_dict_key_set, will_generate_key_set,
                             func_name, set_name, new_data_key)
        # TODO: check all the datasets is either manually created or in
        #       result_dict_key_set
    else:
        check_exact_match_keys(result_dict_key_set, will_generate_key_set,
                               func_name, set_name, new_data_key)


def write_global_feature_h5f(result_dict, global_feature_h5f):
    for key, result in six.viewitems(result_dict):
        if np.isnan(result).any():
            raise ValueError("features {} have nan".format(key))
        with SimpleTimer("Writing generated features {} to hdf5 file"
                         .format(key),
                         end_in_new_line=False):
            if key in global_feature_h5f:
                global_feature_h5f[key][...] = result
            else:
                global_feature_h5f.create_dataset(key, data=result)


def write_data(set_name, result_dict, intermediate_data, global_feature_h5f):
    if set_name == "intermediate_data":
        intermediate_data.update(result_dict)
    elif set_name == "features":
        write_global_feature_h5f(result_dict, global_feature_h5f)
    else:
        raise NotImplementedError()


def generate_data(self, func, set_name, new_data_key, func_name, kwargs):
    with SimpleTimer("Generating {} {} using {}".format(set_name, new_data_key,
                                                        func_name),
                     end_in_new_line=False):  # pylint: disable=C0330
        result_dict = func(self, **kwargs)
    if result_dict is None:
        result_dict = {}
    return result_dict


def will_generate(will_generate_keys, manually_create_dataset=False):
    if isinstance(will_generate_keys, str):
        will_generate_keys = (will_generate_keys,)
    will_generate_key_set = set(will_generate_keys)

    def will_generate_decorator(func):
        # pylint: disable=protected-access
        func._feagen_will_generate_keys = will_generate_keys
        func._feagen_manually_create_dataset = manually_create_dataset

        @wraps(func)
        def func_wrapper(self, set_name, new_data_key, **kwargs):
            func_name = func.__name__
            if manually_create_dataset:
                update_create_dataset_functions(
                    self.global_feature_h5f, will_generate_keys, kwargs)
            result_dict = generate_data(self, func, set_name, new_data_key,
                                        func_name, kwargs)

            check_result_dict_type(result_dict, func_name)
            check_result_dict_keys(result_dict, will_generate_key_set,
                                   func_name, set_name, new_data_key,
                                   manually_create_dataset)
            write_data(set_name, result_dict,
                       self.intermediate_data, self.global_feature_h5f)
        return func_wrapper
    return will_generate_decorator


def will_generate_one_of(will_generate_keys, manually_create_dataset=False):
    if isinstance(will_generate_keys, str):
        will_generate_keys = (will_generate_keys,)

    def will_generate_decorator(func):
        # pylint: disable=protected-access
        func._feagen_will_generate_keys = will_generate_keys
        func._feagen_manually_create_dataset = manually_create_dataset

        @wraps(func)
        def func_wrapper(self, set_name, new_data_key, **kwargs):
            func_name = func.__name__
            if manually_create_dataset:
                update_create_dataset_functions(
                    self.global_feature_h5f, will_generate_keys, kwargs)
            kwargs['will_generate_key'] = new_data_key
            result_dict = generate_data(self, func, set_name, new_data_key,
                                        func_name, kwargs)

            check_result_dict_type(result_dict, func_name)
            check_result_dict_keys(result_dict, set([new_data_key]),
                                   func_name, set_name, new_data_key,
                                   manually_create_dataset)
            write_data(set_name, result_dict,
                       self.intermediate_data, self.global_feature_h5f)
        return func_wrapper
    return will_generate_decorator
