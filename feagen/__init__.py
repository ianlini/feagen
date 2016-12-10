import pkg_resources
from .data_generator import FeatureGenerator
from .bundling import bundle_data

__all__ = ['feature_generator', 'decorators', 'bundling']
__version__ = pkg_resources.get_distribution("feagen").version
