import pkg_resources
from .feature_generator import FeatureGenerator
from .concat_features import save_concat_features

__all__ = ['feature_generator', 'decorators', 'concat_features']
__version__ = pkg_resources.get_distribution("feagen").version
