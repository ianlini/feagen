import os
from os.path import dirname, abspath, join
from tempfile import NamedTemporaryFile, mkdtemp
from shutil import rmtree

import h5py
import yaml
from feagen.tools.feagen_runner import feagen_run_with_configs
from feagen.bundling import flatten_structure


def test_generate_lifetime_features():
    config_dir = join(dirname(abspath(__file__)), '.feagenrc')
    with open(join(config_dir, 'config.yml')) as fp:
        global_config = yaml.load(fp)
    with open(join(config_dir, 'bundle_config.yml')) as fp:
        bundle_config = yaml.load(fp)

    with NamedTemporaryFile(suffix=".h5", delete=False) as fp:
        global_data_hdf_path = fp.name
    data_bundles_dir = mkdtemp()
    data_bundle_hdf_path = join(data_bundles_dir, 'default.h5')

    global_config['global_data_hdf_path'] = global_data_hdf_path
    global_config['data_bundles_dir'] = data_bundles_dir
    global_config['generator_class'] = ("examples.lifetime_prediction."
                                        + global_config['generator_class'])
    csv_path = global_config['generator_kwargs']['data_csv_path']
    global_config['generator_kwargs']['data_csv_path'] = join(
        "examples", "lifetime_prediction", csv_path)

    feagen_run_with_configs(global_config, bundle_config)

    with h5py.File(global_data_hdf_path, "r") as global_data_h5f, \
            h5py.File(data_bundle_hdf_path, "r") as data_bundle_h5f:
        assert (set(global_data_h5f)
                == set(flatten_structure(bundle_config['structure'])))
        assert (set(data_bundle_h5f)
                == {'features', 'test_filters', 'label'})
        assert set(data_bundle_h5f['test_filters']) == {'is_in_test_set'}
        assert data_bundle_h5f['features'].shape == (6, 4)

    os.remove(global_data_hdf_path)
    rmtree(data_bundles_dir)
