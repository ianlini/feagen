from __future__ import print_function
import gc
from functools import wraps, partial

import numpy as np
from bistiming import SimpleTimer


def can_skip(set_name, new_data_name, interm_data, global_feature_h5f,
             show_skip):
    if set_name == "intermediate_data":
        key = new_data_name
        generated_set = interm_data
    elif set_name == "features":
        key = "/" + new_data_name
        generated_set = global_feature_h5f.keys()
    else:
        raise NotImplementedError()

    if key in generated_set:
        if show_skip:
            print("skip {} {}: already generated".format(
                set_name, new_data_name))
        return True
    return False


def data_type(set_name, func_dict, skip_if_exist=True, show_skip=True):
    def intermediate_data_decorator(args):
        func, will_generate_keys = args
        if func.__name__ in func_dict.values():
            raise ValueError("duplicated function name " + func.__name__)
        for key in will_generate_keys:
            if key in func_dict:
                raise ValueError("duplicated {} {} in {} and {}".format(
                    set_name, key, func_dict[key], func.__name__))
            func_dict[key] = func.__name__

        @wraps(func)
        def func_wrapper(self, new_data_name):
            if skip_if_exist and can_skip(
                    set_name, new_data_name, self.intermediate_data,
                    self.global_feature_h5f, show_skip):
                return
            func(self, set_name, new_data_name)
            gc.collect()
        return func_wrapper
    return intermediate_data_decorator


def require_intermediate_data(data_names, interm_data_func_dict):
    if isinstance(data_names, str):
        data_names = (data_names,)

    def require_intermediate_data_decorator(args):
        func, will_generate_keys = args

        @wraps(func)
        def func_wrapper(self, set_name, new_data_name):
            self.require(data_names, interm_data_func_dict)
            data = {key: self.intermediate_data[key] for key in data_names}
            func(self, set_name, new_data_name, data=data)
        return func_wrapper, will_generate_keys
    return require_intermediate_data_decorator


def will_generate(will_generate_keys, manually_create_dataset=False):
    if isinstance(will_generate_keys, str):
        will_generate_keys = (will_generate_keys,)
    will_generate_key_set = set(will_generate_keys)

    def will_generate_decorator(func):
        @wraps(func)
        def func_wrapper(self, set_name, new_data_name, **kwargs):
            if manually_create_dataset:
                kwargs['create_dataset_functions'] = {
                    k: partial(self.global_feature_h5f.create_dataset, k)
                    for k in will_generate_key_set
                }
            with SimpleTimer("Generating {} {} using {}"
                             .format(set_name, new_data_name, func.__name__),
                             end_in_new_line=False):  # pylint: disable=C0330
                result_dict = func(self, **kwargs)
            if result_dict is None:
                result_dict = {}
            if not (hasattr(result_dict, 'keys')
                    and hasattr(result_dict, '__getitem__')):
                raise ValueError("the return value of mehod {} should have "
                                 "keys and __getitem__ methods".format(
                                     func.__name__))

            result_dict_key_set = set(result_dict.keys())
            if manually_create_dataset:
                redundant_key_set = result_dict_key_set - will_generate_key_set
                if len(redundant_key_set) > 0:
                    raise ValueError("the return keys of method {} {} have "
                                     "redundant keys {} while generating {} {}"
                                     .format(
                                         func.__name__, result_dict_key_set,
                                         redundant_key_set, set_name,
                                         new_data_name))
                # TODO: check all the datasets is either manually created or in
                #       result_dict_key_set
            elif will_generate_key_set != result_dict_key_set:
                raise ValueError("the return keys of method {} {} doesn't "
                                 "match {} while generating {} {}".format(
                                     func.__name__, result_dict_key_set,
                                     will_generate_key_set, set_name,
                                     new_data_name))

            if set_name == "intermediate_data":
                self.intermediate_data.update(result_dict)
            elif set_name == "features":
                for key in result_dict_key_set:
                    if np.isnan(result_dict[key]).any():
                        raise ValueError("features {} have nan".format(key))
                    with SimpleTimer("Writing generated features {} "
                                     "to hdf5 file".format(key),
                                     end_in_new_line=False):
                        if key in self.global_feature_h5f:
                            self.global_feature_h5f[key][...] = result_dict[key]
                        else:
                            self.global_feature_h5f.create_dataset(
                                key, data=result_dict[key])
            else:
                raise NotImplementedError()
        return func_wrapper, will_generate_keys
    return will_generate_decorator


intermediate_data = partial(data_type, "intermediate_data")
features = partial(data_type, "features")
