class DataHandler(object):
    pass

class FeaturesHandler(DataHandler):
    pass

class IntermediateDataHandler(DataHandler):
    pass

HANDLER_ALIASES = {
    'features': FeaturesHandler,
    'intermediate_data': IntermediateDataHandler,
}

def get_data_handler(handler):
    if isinstance(handler, str):
        return HANDLER_ALIASES[handler]()
    else:
        return handler
