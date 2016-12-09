from os.path import join, exists

from mkdir_p import mkdir_p


def init_config():
    mkdir_p(".feagenrc")
    default_global_config = """\
generator_class: feature_generator.FeatureGenerator
concat_feature_dir: concat_features
global_feature_hdf_path: global_feature.h5

# The additional arguments that will be given when initiating the feature
# generator object.
generator_kwargs: {}
"""
    default_feature_config = """\
# The name of this feature set. This will be the file name of the concatenated
# data. Another suggested usage is to comment out this line, so the name will be
# obtained from the file name of this config, that is, the name will be the same
# as the config file name without the extension.
name: default

# The structure of the concatenated data. All the involved data will be
# generated and put into the global feature file first (if not exist),
# and then be composed according to this structure and then write to the
# concatenated feature file.
# Only 'features' in the structure dictionary is required and be treated
# differently. The value for the key 'features' can only be list of strings, and
# the features specified in this list will be concatenated to one dataset.
structure:
  id: id
  label: label
  features:
  - feature_1
  - feature_2

# The additional arguments that will be given when initiating the feature
# generator object. If the same key appears in both global config and feature
# config, the value in the feature config will be used.
generator_kwargs: {}
"""
    default_global_config_path = join(".feagenrc", "config.yml")
    if exists(default_global_config_path):
        print("Warning: %s exists so it's not generated."
              % default_global_config_path)
    else:
        with open(default_global_config_path, "w") as fp:
            fp.write(default_global_config)

    default_feature_config_path = join(".feagenrc", "feature_config.yml")
    if exists(default_feature_config_path):
        print("Warning: %s exists so it's not generated."
              % default_feature_config_path)
    else:
        with open(default_feature_config_path, "w") as fp:
            fp.write(default_feature_config)
