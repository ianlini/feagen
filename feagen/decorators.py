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


def will_generate(data_handler, will_generate_keys, mode=None,
                  **handler_kwargs):
    """The decorator that represents what data keys will be generated.

    Parameters
    ==========
    mode: {'full', 'one' or None}
        'full':
            generate all keys in ``will_generate_keys`` in one function call,
            and the node function should return a dict
        'one':
            only generate one of the keys in ``will_generate_keys`` in one
            function call
        None:
            if ``will_generate_keys`` is a string, then the mode is 'one',
            'full' otherwise.
    """
    if isinstance(will_generate_keys, basestring):
        if mode == 'full':
            raise ValueError("mode='full' has no effect if will_generate_keys"
                             "is a string.")
        will_generate_keys = (will_generate_keys,)
        mode = 'one'
    elif mode is None:
        mode = 'full'

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
