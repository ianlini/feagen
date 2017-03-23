from os.path import join
from tempfile import mkdtemp
from shutil import rmtree

import h5py
from feagen.tools.feagen_runner import feagen_run_with_configs


def test_generate_lifetime_features():
    test_output_dir = mkdtemp(prefix="feagen_test_output_")
    h5py_hdf_path = join(test_output_dir, "h5py.h5")
    pandas_hdf_path = join(test_output_dir, "pandas.h5")
    pickle_dir = join(test_output_dir, "pickle")
    data_bundles_dir = join(test_output_dir, "data_bundles")

    global_config = {
        'generator_class': 'feagen.tests.lifetime_feature_generator'
                           '.LifetimeFeatureGenerator',
        'data_bundles_dir': data_bundles_dir,
        'generator_kwargs': {
            'h5py_hdf_path': h5py_hdf_path,
            'pandas_hdf_path': pandas_hdf_path,
            'pickle_dir': pickle_dir,
        },
    }

    bundle_config = {
        'name': 'default',
        'structure': {
            'label': 'lifetime',
            'test_filters': [
                'is_in_test_set',
            ],
            'test_dict': {
                'comparison': [
                    'weight',
                    'height',
                    'mem_raw_data',
                    'pd_weight',
                    'pd_height',
                    'pd_raw_data',
                ],
            },
            'features': [
                'weight',
                'height',
                'mem_raw_data',
                'man_raw_data',
                'pd_weight',
                'pd_height',
                'pd_raw_data',
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
        assert set(data_bundle_h5f) == {'features', 'test_filters', 'label',
                                        'test_dict'}
        assert set(data_bundle_h5f['test_filters']) == {'is_in_test_set'}
        assert set(data_bundle_h5f['test_dict']) == {'comparison'}
        assert (set(data_bundle_h5f['test_dict/comparison'])
                == set(bundle_config['structure']['test_dict']['comparison']))
        assert data_bundle_h5f['features'].shape == (6, 12)

    rmtree(test_output_dir)
