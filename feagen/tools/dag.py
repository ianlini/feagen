import sys
import argparse

import yaml

from .config import get_data_generator_class_from_config


def draw_dag(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Generate DAG.")
    parser.add_argument('-g', '--global-config',
                        default=".feagenrc/config.yml",
                        help="the path of the path configuration YAML file "
                             "(default: .feagenrc/config.yml)")
    parser.add_argument('-d', '--dag-output-path',
                        default="dag.png",
                        help="output image path (default: dag.png)")
    parser.add_argument('-i', '--involved', action='store_true',
                        help="annotate the involved nodes and skipped nodes "
                             "(default: False)")
    args = parser.parse_args(argv)
    with open(args.global_config) as fp:
        global_config = yaml.load(fp)
    generator_class = get_data_generator_class_from_config(global_config)
    generator_class.draw_dag(args.dag_output_path)
