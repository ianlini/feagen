from __future__ import unicode_literals
from io import StringIO

import feagen as fg
from feagen.decorators import (
    require,
    will_generate,
)
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split


class LifetimeFeatureGenerator(fg.FeatureGenerator):

    @will_generate('memory', 'data_df')
    def gen_data_df(self):
        csv = StringIO("""\
id,lifetime,tested_age,weight,height,gender,income
0, 68, 50, 60.1, 170.5, f, 22000
1, 59, 41, 90.4, 168.9, m, 19000
2, 52, 39, 46.2, 173.6, m, 70000
3, 68, 25, 93.9, 180.0, m, 1000000
4, 99, 68, 65.7, 157.6, f, 46000
5, 90, 81, 56.3, 170.2, f, 17000
""")
        return {'data_df': pd.read_csv(csv, index_col='id')}

    @require('data_df')
    @will_generate('h5py', 'label')
    def gen_label(self, data):
        data_df = data['data_df']
        return {'label': data_df['lifetime']}

    @require('data_df')
    @will_generate('h5py', ['weight', 'height'])
    def gen_raw_data_features(self, data):
        data_df = data['data_df']
        return data_df[['weight', 'height']]

    @require('data_df')
    @will_generate('memory', 'mem_raw_data')
    def gen_mem_raw_data(self, data):
        data_df = data['data_df']
        return {'mem_raw_data': data_df[['weight', 'height']].values}

    @require('data_df')
    @will_generate('h5py', 'man_raw_data', manually_create_dataset=True)
    def gen_man_raw_data(self, data, create_dataset_functions):
        data_df = data['data_df']
        dset = create_dataset_functions['man_raw_data'](
            shape=(data_df.shape[0], 2))
        dset[...] = data_df[['weight', 'height']].values

    @require('data_df')
    @will_generate('pandas_hdf', ['pd_weight', 'pd_height'])
    def gen_raw_data_table(self, data):
        data_df = data['data_df']
        result_df = data_df.loc[:, ['weight', 'height']]
        result_df.rename(columns={'weight': 'pd_weight', 'height': 'pd_height'},
                         inplace=True)
        return result_df

    @require('data_df')
    @will_generate('pandas_hdf', 'pd_raw_data')
    def gen_raw_data_df(self, data):
        data_df = data['data_df']
        return {'pd_raw_data': data_df[['weight', 'height']]}

    @require('data_df')
    @will_generate('h5py', 'BMI')
    def gen_bmi(self, data):
        data_df = data['data_df']
        bmi = data_df['weight'] / ((data_df['height'] / 100) ** 2)
        return {'BMI': bmi}

    @require('data_df')
    @will_generate('h5py', r'\w+_divided_by_\w+', mode='one')
    def gen_divided_by(self, will_generate_key, data):
        import re
        data_df = data['data_df']
        matched = re.match(r"(?P<data1>\w+)_divided_by_(?P<data2>\w+)",
                           will_generate_key)
        division_result = (data_df[matched.group('data1')]
                           / data_df[matched.group('data2')])
        return {will_generate_key: division_result}

    @require('data_df')
    @will_generate('pickle', 'train_test_split')
    def gen_train_test_split(self, data):
        data_df = data['data_df']
        train_id, test_id = train_test_split(
            data_df.index, test_size=0.5, random_state=0)
        return {'train_test_split': (train_id, test_id)}

    @require(('data_df', 'train_test_split'))
    @will_generate('h5py', 'is_in_test_set')
    def gen_is_in_test_set(self, data):
        data_df = data['data_df']
        _, test_id = data['train_test_split']
        is_in_test_set = data_df.index.isin(test_id)
        sparse_is_in_test_set = csr_matrix(is_in_test_set[:, np.newaxis])
        return {'is_in_test_set': sparse_is_in_test_set}
