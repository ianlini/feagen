from __future__ import print_function
import gc
from functools import wraps, partial


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


def data_type(set_name, skip_if_exist=True, show_skip=True):
    def data_type_decorator(func):
        # pylint: disable=protected-access
        if func._feagen_manually_create_dataset and set_name != "features":
            raise ValueError("Only features can use manually_create_dataset"
                             "=True, but it's used in %s." % func.__name__)
        func._feagen_data_type = set_name

        @wraps(func)
        def func_wrapper(self, new_data_name):
            if skip_if_exist and can_skip(
                    set_name, new_data_name, self.intermediate_data,
                    self.global_feature_h5f, show_skip):
                return
            func(self, set_name, new_data_name)
            gc.collect()
        return func_wrapper
    return data_type_decorator


intermediate_data = partial(data_type, "intermediate_data")
features = partial(data_type, "features")
