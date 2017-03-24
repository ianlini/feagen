
class PandasHDFDataset(object):
    """h5py Dataset-like wrapper for pandas HDFStore."""

    def __init__(self, hdf_store, key):
        self._hdf_store = hdf_store
        self.key = key
        storer = self._hdf_store.get_storer(key)
        if isinstance(storer.shape, tuple):
            self.shape = storer.shape
        else:
            self.shape = (storer.nrows, storer.ncols)
            assert (storer.nrows is not None) and (storer.ncols is not None)

    @property
    def value(self):
        return self._hdf_store[self.key]

    @property
    def dtype(self):
        return self._hdf_store.select(self.key, start=0, stop=1).values.dtype

    def select(self, *arg, **kwargs):
        return self._hdf_store.select(self.key, *arg, **kwargs)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._hdf_store.select(self.key, start=key, stop=key + 1)
        elif isinstance(key, slice):
            if key.step is None:
                return self._hdf_store.select(self.key,
                                              start=key.start, stop=key.stop)
        raise NotImplementedError("Key {} is not supported".format(key))
