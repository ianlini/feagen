import sys
import argparse

import yaml

from .config import (
    get_data_generator_class_from_config,
    get_data_generator_from_config,
)
from ..bundling import get_data_keys_from_structure


def draw_dag(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Generate DAG.")
    parser.add_argument('-g', '--global-config',
                        default=".feagenrc/config.yml",
                        help="the path of the path configuration YAML file "
                             "(default: .feagenrc/config.yml)")
    parser.add_argument('-b', '--bundle-config',
                        default=".feagenrc/bundle_config.yml",
                        help="the path of the bundle configuration YAML file "
                             "(default: .feagenrc/bundle_config.yml)")
    parser.add_argument('-d', '--dag-output-path',
                        default="dag.png",
                        help="output image path (default: dag.png)")
    parser.add_argument('-i', '--involved', action='store_true',
                        help="annotate the involved nodes and skipped nodes "
                             "(default: False)")
    args = parser.parse_args(argv)
    with open(args.global_config) as fp:
        global_config = yaml.load(fp)
    with open(args.bundle_config) as fp:
        bundle_config = yaml.load(fp)
    data_keys = get_data_keys_from_structure(bundle_config['structure'])
    if args.involved:
        data_generator = get_data_generator_from_config(global_config)
        data_generator.draw_involved_dag(args.dag_output_path, data_keys)
    else:
        generator_class = get_data_generator_class_from_config(global_config)
        generator_class.draw_dag(args.dag_output_path, data_keys)
