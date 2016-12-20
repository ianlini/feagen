from os.path import join, exists

from mkdir_p import mkdir_p


def init_config():
    mkdir_p(".feagenrc")
    default_global_config = """\
generator_class: feature_generator.FeatureGenerator
data_bundles_dir: data_bundles

# The additional arguments that will be given when initiating the data generator
# object.
generator_kwargs:
  global_data_hdf_path:
    global_data.h5
"""
    default_bundle_config = """\
# The name of this bundle. This will be the file name of the data bundle.
# Another suggested usage is to comment out this line, so the name will be
# obtained from the file name of this config, that is, the name will be the same
# as the config file name without the extension.
name: default

# The structure of the data bundle. All the involved data will be generated and
# put into the global data file first (if data not exist), and then be bundled
# according to this structure, and then write to the data bundle file.
structure:
  id: id
  label: label
  features:
  - feature_1
  - feature_2

# Special configuration for the structure. Here we set concat=True for
# 'features'. It means that the data list in 'features' will be concatenated
# into a dataset.
structure_config:
  features:
    concat: True
"""
    default_global_config_path = join(".feagenrc", "config.yml")
    if exists(default_global_config_path):
        print("Warning: %s exists so it's not generated."
              % default_global_config_path)
    else:
        with open(default_global_config_path, "w") as fp:
            fp.write(default_global_config)

    default_bundle_config_path = join(".feagenrc", "bundle_config.yml")
    if exists(default_bundle_config_path):
        print("Warning: %s exists so it's not generated."
              % default_bundle_config_path)
    else:
        with open(default_bundle_config_path, "w") as fp:
            fp.write(default_bundle_config)
