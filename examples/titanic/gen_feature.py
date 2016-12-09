
import os

import pandas as pd

import feagen as fg
from feagen.decorators import (
    will_generate,
    require,
)

class TitanicFeatureGenerator(fg.FeatureGenerator):

    def __init__(self, global_feature_hdf_path, data_train_csv_path, data_test_csv_path):
        super(TitanicFeatureGenerator, self).__init__(global_feature_hdf_path)
        self.data_train_csv_path = data_train_csv_path
        self.data_test_csv_path = data_test_csv_path

    @will_generate('intermediate_data', 'data_df')
    def gen_data_df(self):
        trn_df = pd.read_csv(self.data_train_csv_path, index_col='PassengerId')
        trn_df['is_test'] = 0
        tst_df = pd.read_csv(self.data_test_csv_path, index_col='PassengerId')
        tst_df['is_test'] = 1
        tst_df['Survived'] = -1
        return {'data_df': pd.concat([trn_df, tst_df])}

    @require('data_df')
    @will_generate('features', 'passenger_id')
    def gen_passenger_id(self, data):
        data_df = data['data_df']
        return {'passenger_id': data_df.index.values}

    @require('data_df')
    @will_generate('features', 'is_test')
    def gen_is_test(self, data):
        return {'is_test': data['data_df']['is_test'].values}

    @require('data_df')
    @will_generate('features', 'label')
    def gen_label(self, data):
        data_df = data['data_df']
        return {'label': data_df['Survived']}

    @require('data_df')
    @will_generate('features', 'pclass')
    def gen_pclass(self, data):
        from sklearn.preprocessing import OneHotEncoder
        data_df = data['data_df']
        return {'pclass': OneHotEncoder().fit_transform(data_df['pclass'].values)}

    @require('data_df')
    @will_generate('features', 'family_size')
    def gen_family_size(self, data):
        data_df = data['data_df']
        return {'family_size': (data_df['SibSp'] + data_df['Parch']).values}

    @require('data_df')
    @will_generate('features', ['Parch', 'SibSp'])
    def gen_parch_sibsp(self, data):
        data_df = data['data_df']
        return data_df[['Parch', 'SibSp']]

    @require('data_df')
    @will_generate('features', 'is_validation')
    def gen_is_validation(self, data):
        from sklearn.model_selection import train_test_split
        import numpy as np
        data_df = data['data_df']
        df = pd.DataFrame(0, index=data_df.index, columns=['is_validation'], dtype=bool)
        random_state = np.random.RandomState(1126)
        _, valid_id = train_test_split(
            data_df.index, test_size=0.3, random_state=random_state)
        df.loc[valid_id, 'is_validation'] = 1
        return df


def main():
    data_csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    concat_feature_hdf_path = os.path.join(data_csv_dir, 'concat_feature.h5')
    global_feature_hdf_path = os.path.join(data_csv_dir, 'data.h5')
    feature_list = ['family_size', 'Parch', 'SibSp']
    label_list = ['label']
    info_list = ['is_validation', 'is_test', 'passenger_id']
    generator = TitanicFeatureGenerator(global_feature_hdf_path,
        os.path.join(data_csv_dir, 'train.csv'),
        os.path.join(data_csv_dir, 'test.csv'))
    generator.generate(feature_list + label_list + info_list)

    fg.save_concat_features(
        feature_list=feature_list,
        global_feature_hdf_path=global_feature_hdf_path,
        concat_feature_hdf_path=concat_feature_hdf_path,
        extra_data={'label': label_list,
            'passenger_id': ['passenger_id'],
            'test_filter_list': ['is_test'],
            'valid_filter_list': ['is_validation']})


if __name__ == '__main__':
    main()
