from os.path import join
from tempfile import mkdtemp
import unittest
from shutil import rmtree

import pandas as pd
from feagen.data_wrappers.pandas_hdf import get_shape_from_pandas_hdf_storer


class Test(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = mkdtemp(prefix="feagen_test_output_")
        pandas_hdf_path = join(self.test_output_dir, "pandas.h5")
        self.hdf_store = pd.HDFStore(pandas_hdf_path)

    def tearDown(self):
        self.hdf_store.close()
        rmtree(self.test_output_dir)

    def test_get_shape_from_pandas_hdf_storer_df(self):
        idx = [1, 2, 3, 5, 4]
        col = [10, 9, 6, 7]
        df = pd.DataFrame(0, index=idx, columns=col)
        self.hdf_store['test'] = df
        shape = get_shape_from_pandas_hdf_storer(
            self.hdf_store.get_storer('test'))
        assert shape == (5, 4)

    def test_get_shape_from_pandas_hdf_storer_df_table(self):
        idx = [1, 2, 3, 5, 4]
        col = [10, 9, 6, 7]
        df = pd.DataFrame(0, index=idx, columns=col)
        self.hdf_store.put('test', df, format='table')
        shape = get_shape_from_pandas_hdf_storer(
            self.hdf_store.get_storer('test'))
        assert shape == (5, 4)

    def test_get_shape_from_pandas_hdf_storer_df_m_idx(self):
        idx = pd.MultiIndex.from_product([[0, 1], [0, 1, 2]])
        col = [10, 9, 6, 7]
        df = pd.DataFrame(0, index=idx, columns=col)
        self.hdf_store['test'] = df
        shape = get_shape_from_pandas_hdf_storer(
            self.hdf_store.get_storer('test'))
        assert shape == (6, 4)

    def test_get_shape_from_pandas_hdf_storer_df_m_idx_table(self):
        idx = pd.MultiIndex.from_product([[0, 1], [0, 1, 2]])
        col = [10, 9, 6, 7]
        df = pd.DataFrame(0, index=idx, columns=col)
        self.hdf_store.put('test', df, format='table')
        shape = get_shape_from_pandas_hdf_storer(
            self.hdf_store.get_storer('test'))
        assert shape == (6, 4)

    def test_get_shape_from_pandas_hdf_storer_df_m_col(self):
        idx = [10, 9, 6, 7]
        col = pd.MultiIndex.from_product([[0, 1], [0, 1, 2]])
        df = pd.DataFrame(0, index=idx, columns=col)
        self.hdf_store['test'] = df
        shape = get_shape_from_pandas_hdf_storer(
            self.hdf_store.get_storer('test'))
        # TODO: change to (4, 6)
        assert shape is None

    def test_get_shape_from_pandas_hdf_storer_df_m_col_table(self):
        idx = [10, 9, 6, 7]
        col = pd.MultiIndex.from_product([[0, 1], [0, 1, 2]])
        df = pd.DataFrame(0, index=idx, columns=col)
        self.hdf_store.put('test', df, format='table')
        shape = get_shape_from_pandas_hdf_storer(
            self.hdf_store.get_storer('test'))
        assert shape == (4, 6)

    def test_get_shape_from_pandas_hdf_storer_df_m_idx_m_col(self):
        idx = pd.MultiIndex.from_product([[0, 1], [0, 1, 2]])
        col = pd.MultiIndex.from_product([[0, 1], [0, 1]])
        df = pd.DataFrame(0, index=idx, columns=col)
        self.hdf_store['test'] = df
        shape = get_shape_from_pandas_hdf_storer(
            self.hdf_store.get_storer('test'))
        # TODO: change to (6, 4)
        assert shape is None

    def test_get_shape_from_pandas_hdf_storer_s(self):
        idx = [0, 2, 1, 4, 3]
        s = pd.Series(0, index=idx)
        self.hdf_store['test'] = s
        shape = get_shape_from_pandas_hdf_storer(
            self.hdf_store.get_storer('test'))
        assert shape == (5,)

    def test_get_shape_from_pandas_hdf_storer_s_table(self):
        idx = [0, 2, 1, 4, 3]
        s = pd.Series(0, index=idx)
        self.hdf_store.put('test', s, format='table')
        shape = get_shape_from_pandas_hdf_storer(
            self.hdf_store.get_storer('test'))
        assert shape == (5,)

    def test_get_shape_from_pandas_hdf_storer_s_m_idx(self):
        idx = pd.MultiIndex.from_product([[0, 1], [0, 1, 2]])
        s = pd.Series(0, index=idx)
        self.hdf_store['test'] = s
        shape = get_shape_from_pandas_hdf_storer(
            self.hdf_store.get_storer('test'))
        assert shape == (6,)

    def test_get_shape_from_pandas_hdf_storer_s_m_idx_table(self):
        idx = pd.MultiIndex.from_product([[0, 1], [0, 1, 2]])
        s = pd.Series(0, index=idx)
        self.hdf_store.put('test', s, format='table')
        shape = get_shape_from_pandas_hdf_storer(
            self.hdf_store.get_storer('test'))
        assert shape == (6,)
