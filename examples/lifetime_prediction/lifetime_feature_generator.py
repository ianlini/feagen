import pandas as pd
from sklearn.model_selection import train_test_split
import feagen as fg
from feagen.decorators import (
    will_generate,
    will_generate_one_of,
    require,
)


class LifetimeFeatureGenerator(fg.FeatureGenerator):

    def __init__(self, global_data_hdf_path, data_csv_path):
        super(LifetimeFeatureGenerator, self).__init__(global_data_hdf_path)
        self.data_csv_path = data_csv_path

    @will_generate('memory', 'data_df')
    def gen_data_df(self):
        return {'data_df': pd.read_csv(self.data_csv_path, index_col='id')}

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
