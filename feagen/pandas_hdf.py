def get_shape_from_pandas_hdf_storer(storer):
    if not storer.is_table:
        shape = storer.shape
    else:
        if storer.is_multi_index:
            ncols = storer.ncols - len(storer.levels)
        else:
            ncols = storer.ncols
        shape = (storer.nrows, ncols)
        assert (shape[0] is not None) and (shape[1] is not None)
    return shape


class PandasHDFDataset(object):
    """h5py Dataset-like wrapper for pandas HDFStore."""

    def __init__(self, hdf_store, key):
        self._hdf_store = hdf_store
        self._storer = hdf_store.get_storer(key)
        self.key = key
        self.shape = get_shape_from_pandas_hdf_storer(self._storer)

    @property
    def value(self):
        return self._hdf_store[self.key]

    @property
    def dtype(self):
        return self._hdf_store.select(self.key, start=0, stop=1).values.dtype

    def select(self, *arg, **kwargs):
        return self._hdf_store.select(self.key, *arg, **kwargs)

    def select_column(self, *arg, **kwargs):
        return self._hdf_store.select_column(self.key, *arg, **kwargs)

    def select_as_coordinates(self, *arg, **kwargs):
        return self._hdf_store.select_as_coordinates(self.key, *arg, **kwargs)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._hdf_store.select(self.key, start=key, stop=key + 1)
        elif isinstance(key, slice):
            if key.step is None:
                return self._hdf_store.select(self.key,
                                              start=key.start, stop=key.stop)
        raise NotImplementedError("Key {} is not supported".format(key))
