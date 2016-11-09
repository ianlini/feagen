from functools import partial

from feagen.decorators import (require_intermediate_data,
                               intermediate_data, features)


FEATURE_FUNC_DICT = {}
INTERMEDIATE_DATA_FUNC_DICT = {}
require_intermediate_data = partial(
    require_intermediate_data,
    interm_data_func_dict=INTERMEDIATE_DATA_FUNC_DICT)
intermediate_data = partial(
    intermediate_data, func_dict=INTERMEDIATE_DATA_FUNC_DICT)
features = partial(features, func_dict=FEATURE_FUNC_DICT)
