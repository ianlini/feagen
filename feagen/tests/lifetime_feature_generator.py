from __future__ import unicode_literals
from io import StringIO

import pandas as pd
from sklearn.model_selection import train_test_split
import feagen as fg
from feagen.decorators import (
    will_generate,
    will_generate_one_of,
    require,
)


class LifetimeFeatureGenerator(fg.FeatureGenerator):

    def __init__(self, global_data_hdf_path):
        super(LifetimeFeatureGenerator, self).__init__(global_data_hdf_path)

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
    @will_generate('h5py', 'BMI')
    def gen_bmi(self, data):
        data_df = data['data_df']
        bmi = data_df['weight'] / ((data_df['height'] / 100) ** 2)
        return {'BMI': bmi}

    @require('data_df')
    @will_generate_one_of('h5py', r'\w+_divided_by_\w+')
    def gen_divided_by(self, will_generate_key, data):
        import re
        data_df = data['data_df']
        matched = re.match(r"(?P<data1>\w+)_divided_by_(?P<data2>\w+)",
                           will_generate_key)
        division_result = (data_df[matched.group('data1')]
                           / data_df[matched.group('data2')])
        return {will_generate_key: division_result}

    @require('data_df')
    @will_generate('h5py', 'is_in_test_set')
    def gen_is_in_test_set(self, data):
        data_df = data['data_df']
        _, test_id = train_test_split(
            data_df.index, test_size=0.5, random_state=0)
        is_in_test_set = data_df.index.isin(test_id)
        return {'is_in_test_set': is_in_test_set}
