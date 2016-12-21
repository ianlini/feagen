from os.path import basename, splitext, join
import sys
import argparse
from importlib import import_module
import collections

import yaml
from mkdir_p import mkdir_p
from feagen.dag import draw_dag

from ..bundling import get_data_keys_from_structure


def feagen_run_with_configs(global_config, bundle_config, dag_output_path=None,
                            no_bundle=False):
    # TODO: check the config
    """Generate feature with configurations.

    global_config (collections.Mapping): global configuration
        generator_class: string
        data_bundles_dir: string
        generator_kwargs: collections.Mapping

    bundle_config (collections.Mapping): bundle configuration
        name: string
        structure: collections.Mapping
    """
    if not isinstance(global_config, collections.Mapping):
        raise ValueError("global_config should be a "
                         "collections.Mapping object.")
    if not isinstance(bundle_config, collections.Mapping):
        raise ValueError("bundle_config should be a "
                         "collections.Mapping object.")

    module_name, class_name = global_config['generator_class'].rsplit(".", 1)
    module = import_module(module_name)
    generator_class = getattr(module, class_name)
    generator_kwargs = global_config['generator_kwargs']
    data_generator = generator_class(**generator_kwargs)

    data_keys = get_data_keys_from_structure(bundle_config['structure'])
    involved_dag = data_generator.generate(data_keys)

    if dag_output_path is not None:
        draw_dag(involved_dag, dag_output_path)

    if not no_bundle:
        mkdir_p(global_config['data_bundles_dir'])
        bundle_path = join(global_config['data_bundles_dir'],
                           bundle_config['name'] + '.h5')
        data_generator.bundle(
            bundle_config['structure'], data_bundle_hdf_path=bundle_path,
            structure_config=bundle_config['structure_config'])


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
    parser.add_argument('--no-bundle', action='store_true',
                        help="not generate the data bundle")
    args = parser.parse_args(argv)
    with open(args.global_config) as fp:
        global_config = yaml.load(fp)
    with open(args.bundle_config) as fp:
        bundle_config = yaml.load(fp)
    filename_without_extension = splitext(basename(args.bundle_config))[0]
    bundle_config.setdefault('name', filename_without_extension)
    feagen_run_with_configs(global_config, bundle_config, args.dag_output_path,
                            args.no_bundle)
