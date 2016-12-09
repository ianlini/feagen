from os.path import basename, splitext
import sys
import argparse
from importlib import import_module

import yaml

from ..concat_features import flatten_structure


def feagen_run_with_configs(global_config, feature_config):
    module_name, class_name = global_config['generator_class'].rsplit(".", 1)
    module = import_module(module_name)
    generator_class = getattr(module, class_name)
    generator_kwargs = global_config['generator_kwargs']
    generator_kwargs.update(feature_config['generator_kwargs'])
    feature_generator = generator_class(
        global_feature_hdf_path=global_config['global_feature_hdf_path'],
        **generator_kwargs)

    data_keys = flatten_structure(feature_config['structure'])
    involved_dag = feature_generator.generate(data_keys)
    import ipdb; ipdb.set_trace()

    fg.save_concat_features(
        feature_list=feature_list,
        global_feature_hdf_path=global_feature_hdf_path,
        concat_feature_hdf_path=concat_feature_hdf_path,
        extra_data={'label': label_list, 'test_filter_list': test_filter_list})


def feagen_run(argv=sys.argv[1:]):

    parser = argparse.ArgumentParser(description='Generate features.')
    parser.add_argument('-g', '--global-config',
                        default=".feagenrc/config.yml",
                        help="the path of the path configuration yaml file")
    parser.add_argument('-f', '--feature-config',
                        default=".feagenrc/feature_config.yml",
                        help="the path of the feature configuration yaml file")
    args = parser.parse_args(argv)
    with open(args.global_config) as fp:
        global_config = yaml.load(fp)
    with open(args.feature_config) as fp:
        feature_config = yaml.load(fp)
    filename_without_extension = splitext(basename(args.feature_config))[0]
    feature_config.setdefault('name', filename_without_extension)
    # TODO: check the config
    feagen_run_with_configs(global_config, feature_config)
