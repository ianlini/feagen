from os.path import abspath, dirname, join
from shutil import rmtree
from tempfile import mkdtemp

from feagen.bundling import get_data_keys_from_structure
from feagen.tools.feagen_runner import feagen_run_with_configs
import h5py
import yaml


def test_generate_lifetime_features():
    config_dir = join(dirname(abspath(__file__)), '.feagenrc')
    with open(join(config_dir, 'config.yml')) as fp:
        global_config = yaml.load(fp)
    with open(join(config_dir, 'bundle_config.yml')) as fp:
        bundle_config = yaml.load(fp)

    test_output_dir = mkdtemp(prefix="feagen_test_output_")
    h5py_hdf_path = join(test_output_dir, "h5py.h5")
    data_bundles_dir = join(test_output_dir, "data_bundles")

    global_config['data_bundles_dir'] = data_bundles_dir
    global_config['generator_class'] = ("examples.lifetime_prediction."
                                        + global_config['generator_class'])
    csv_path = global_config['generator_kwargs']['data_csv_path']
    global_config['generator_kwargs']['data_csv_path'] = join(
        "examples", "lifetime_prediction", csv_path)
    global_config['generator_kwargs']['h5py_hdf_path'] = \
        h5py_hdf_path

    feagen_run_with_configs(global_config, bundle_config)

    data_bundle_hdf_path = join(data_bundles_dir, 'default.h5')
    with h5py.File(h5py_hdf_path, "r") as global_data_h5f, \
            h5py.File(data_bundle_hdf_path, "r") as data_bundle_h5f:
        assert (set(global_data_h5f)
                == set(get_data_keys_from_structure(
                    bundle_config['structure'])))
        assert set(data_bundle_h5f) == {'features', 'test_filters', 'label'}
        assert set(data_bundle_h5f['test_filters']) == {'is_in_test_set'}
        assert data_bundle_h5f['features'].shape == (6, 4)

    rmtree(test_output_dir)
