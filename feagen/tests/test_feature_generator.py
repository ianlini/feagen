import os
from os.path import join
from tempfile import mkstemp, mkdtemp
from shutil import rmtree

import h5py
from feagen.tools.feagen_runner import feagen_run_with_configs
from feagen.bundling import get_data_keys_from_structure


def test_generate_lifetime_features():
    global_data_hdf_path = mkstemp(suffix=".h5")[1]
    data_bundles_dir = mkdtemp()

    global_config = {
        'generator_class': 'feagen.tests.lifetime_feature_generator'
                           '.LifetimeFeatureGenerator',
        'data_bundles_dir': data_bundles_dir,
        'generator_kwargs': {
            'global_data_hdf_path': global_data_hdf_path,
        },
    }

    bundle_config = {
        'name': 'default',
        'structure': {
            'label': 'label',
            'test_filters': [
                'is_in_test_set',
            ],
            'features': [
                'weight',
                'mem_weight',
                'height',
                'BMI',
                'weight_divided_by_height',
            ],
        },
        'structure_config': {
            'features': {
                'concat': True,
            }
        }
    }

    feagen_run_with_configs(global_config, bundle_config)

    data_bundle_hdf_path = join(data_bundles_dir, bundle_config['name'] + '.h5')
    with h5py.File(data_bundle_hdf_path, "r") as data_bundle_h5f:
        assert set(data_bundle_h5f) == {'features', 'test_filters', 'label'}
        assert set(data_bundle_h5f['test_filters']) == {'is_in_test_set'}
        assert (data_bundle_h5f['features'].shape
                == (6, len(bundle_config['structure']['features'])))

    os.remove(global_data_hdf_path)
    rmtree(data_bundles_dir)
