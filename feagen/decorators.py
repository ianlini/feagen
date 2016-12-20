from past.builtins import basestring


def require(data_keys):
    if isinstance(data_keys, basestring):
        data_keys = (data_keys,)

    def require_decorator(func):
        # pylint: disable=protected-access
        if not hasattr(func, '_feagen_require'):
            func._feagen_require = []
        func._feagen_require.extend(data_keys)
        return func
    return require_decorator


def will_generate(data_handler, will_generate_keys, **handler_kwargs):
    if isinstance(will_generate_keys, basestring):
        will_generate_keys = (will_generate_keys,)

    def will_generate_decorator(func):
        # pylint: disable=protected-access
        if hasattr(func, '_feagen_will_generate'):
            raise NotImplementedError("Multiple will_generate is not supported"
                                      "yet.")
        func._feagen_will_generate = {
            'mode': 'full',
            'handler': data_handler,
            'keys': will_generate_keys,
            'handler_kwargs': handler_kwargs,
        }
        return func
    return will_generate_decorator


def will_generate_one_of(data_handler, will_generate_keys, **handler_kwargs):
    if isinstance(will_generate_keys, basestring):
        will_generate_keys = (will_generate_keys,)

    def will_generate_one_of_decorator(func):
        # pylint: disable=protected-access
        if hasattr(func, '_feagen_will_generate'):
            raise NotImplementedError("Multiple will_generate is not supported"
                                      "yet.")
        func._feagen_will_generate = {
            'mode': 'one',
            'handler': data_handler,
            'keys': will_generate_keys,
            'handler_kwargs': handler_kwargs,
        }
        return func
    return will_generate_one_of_decorator
