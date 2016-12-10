import pkg_resources
from .data_generator import FeatureGenerator
from .bundling import bundle_data

__all__ = ['tools', 'bundling', 'data_generator', 'data_handler' 'decorators']
__version__ = pkg_resources.get_distribution("feagen").version
