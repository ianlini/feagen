import sys
import argparse


def feagen_run(argv=sys.argv[1:]):

    parser = argparse.ArgumentParser(description='Generate features.')
    parser.add_argument('-p', '--path-config',
                        default=".feagenrc/path_config.yml",
                        help="the path of the path configuration yaml file")
    parser.add_argument('-f', '--feature-config',
                        default=".feagenrc/feature_config.yml",
                        help="the path of the feature configuration yaml file")
    args = parser.parse_args(argv)
    print(args)
