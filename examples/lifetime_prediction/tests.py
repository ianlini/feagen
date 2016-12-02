import os

from tempfile import NamedTemporaryFile
from .generate_lifetime_features import generate_lifetime_features


def test_generate_lifetime_features():
    with NamedTemporaryFile(suffix=".h5", delete=False) as fp:
        global_feature_hdf_path = fp.name
    with NamedTemporaryFile(suffix=".h5", delete=False) as fp:
        concat_feature_hdf_path = fp.name
    generate_lifetime_features(global_feature_hdf_path, concat_feature_hdf_path)
    os.remove(global_feature_hdf_path)
    os.remove(concat_feature_hdf_path)
