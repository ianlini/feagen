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


def will_generate(data_handler, will_generate_keys, mode='full',
                  **handler_kwargs):
    """

    Parameters
    ==========
    mode: {'full', 'one'}
        full: generate all keys in ``will_generate_keys`` in one function call
        one: only generate one of the keys in ``will_generate_keys`` in one
             function call
    """
    if isinstance(will_generate_keys, basestring):
        will_generate_keys = (will_generate_keys,)

    def will_generate_decorator(func):
        # pylint: disable=protected-access
        if hasattr(func, '_feagen_will_generate'):
            raise NotImplementedError("Multiple will_generate is not supported"
                                      "yet.")
        func._feagen_will_generate = {
            'mode': mode,
            'handler': data_handler,
            'keys': will_generate_keys,
            'handler_kwargs': handler_kwargs,
        }
        return func
    return will_generate_decorator
