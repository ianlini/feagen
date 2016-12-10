import sys
import argparse
from importlib import import_module

import yaml


def draw_full_dag(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Generate DAG.")
    parser.add_argument('-g', '--global-config',
                        default=".feagenrc/config.yml",
                        help="the path of the path configuration YAML file "
                             "(default: .feagenrc/config.yml)")
    parser.add_argument('-d', '--dag-output-path',
                        default="dag.png",
                        help="output image path (default: dag.png)")
    args = parser.parse_args(argv)
    with open(args.global_config) as fp:
        global_config = yaml.load(fp)
    # TODO: check the config
    module_name, class_name = global_config['generator_class'].rsplit(".", 1)
    module = import_module(module_name)
    generator_class = getattr(module, class_name)
    generator_class.draw_dag(args.dag_output_path)
