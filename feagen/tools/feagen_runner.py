from os.path import basename, splitext, join
import sys
import argparse
from importlib import import_module

import yaml
from mkdir_p import mkdir_p
from feagen.data_generator import draw_dag

from ..bundling import flatten_structure, bundle_data


def feagen_run_with_configs(global_config, bundle_config, dag_output_path=None):
    # TODO: check the config
    """Generate feature with configurations.

    global_config (dict): global configuration
        generator_class: string
        generator_kwargs: dict

    bundle_config (dict): bundle configuration
        name: string
        structure: dict
    """
    if not isinstance(global_config, dict):
        raise ValueError("global_config should be a dictionary")
    if not isinstance(bundle_config, dict):
        raise ValueError("global_config should be a dictionary")

    module_name, class_name = global_config['generator_class'].rsplit(".", 1)
    module = import_module(module_name)
    generator_class = getattr(module, class_name)
    generator_kwargs = global_config['generator_kwargs']
    data_generator = generator_class(**generator_kwargs)

    data_keys = flatten_structure(bundle_config['structure'])
    involved_dag = data_generator.generate(data_keys)
    if dag_output_path is not None:
        draw_dag(involved_dag, dag_output_path)

    mkdir_p(global_config['data_bundles_dir'])
    bundle_path = join(global_config['data_bundles_dir'],
                       bundle_config['name'] + '.h5')
    bundle_data(bundle_config['structure'],
                global_data_hdf_path=generator_kwargs['global_data_hdf_path'],
                data_bundle_hdf_path=bundle_path)


def feagen_run(argv=sys.argv[1:]):

    parser = argparse.ArgumentParser(
        description="Generate global data and data bundle.")
    parser.add_argument('-g', '--global-config',
                        default=".feagenrc/config.yml",
                        help="the path of the path configuration YAML file "
                             "(default: .feagenrc/config.yml)")
    parser.add_argument('-b', '--bundle-config',
                        default=".feagenrc/bundle_config.yml",
                        help="the path of the bundle configuration YAML file "
                             "(default: .feagenrc/bundle_config.yml)")
    parser.add_argument('-d', '--dag-output-path', default=None,
                        help="draw the involved subDAG to the provided path "
                             "(default: None)")
    args = parser.parse_args(argv)
    with open(args.global_config) as fp:
        global_config = yaml.load(fp)
    with open(args.bundle_config) as fp:
        bundle_config = yaml.load(fp)
    filename_without_extension = splitext(basename(args.bundle_config))[0]
    bundle_config.setdefault('name', filename_without_extension)
    feagen_run_with_configs(global_config, bundle_config, args.dag_output_path)
