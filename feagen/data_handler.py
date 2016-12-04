class DataHandler(object):
    @property
    def data_type(self):
        return self.__class__.__name__

class HDF5FeatureHandler(DataHandler):
    @property
    def data_type(self):
        return 'hdf5_feature'

class MemoryIntermediateDataHandler(DataHandler):
    @property
    def data_type(self):
        return 'memory_intermediate_data'

HANDLER_ALIASES = {
    'features': HDF5FeatureHandler,
    'intermediate_data': MemoryIntermediateDataHandler,
}

def get_data_handler(handler):
    if isinstance(handler, str):
        return HANDLER_ALIASES[handler]()
    else:
        return handler
