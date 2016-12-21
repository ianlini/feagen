import pkg_resources
from .data_generator import FeatureGenerator

__all__ = ['tools', 'bundling', 'data_generator', 'data_handlers', 'decorators',
           'dag']
__version__ = pkg_resources.get_distribution("feagen").version
