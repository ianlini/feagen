
import os

import numpy as np
import pandas as pd

import feagen as fg
from feagen.decorators import (
    will_generate,
    require,
)

class TitanicFeatureGenerator(fg.FeatureGenerator):

    def __init__(self, h5py_hdf_path, data_train_csv_path, data_test_csv_path):
        super(TitanicFeatureGenerator, self).__init__(h5py_hdf_path)
        self.data_train_csv_path = data_train_csv_path
        self.data_test_csv_path = data_test_csv_path

    @will_generate('memory', 'data_df')
    def gen_data_df(self):
        trn_df = pd.read_csv(self.data_train_csv_path, index_col='PassengerId')
        tst_df = pd.read_csv(self.data_test_csv_path, index_col='PassengerId')
        tst_df['Survived'] = -1
        return {'data_df': pd.concat([trn_df, tst_df])}

    @require('data_df')
    @will_generate('h5py', 'passenger_id')
    def gen_passenger_id(self, data):
        data_df = data['data_df']
        return {'passenger_id': data_df.index.values}

    @require('data_df')
    @will_generate('h5py', 'is_test')
    def gen_is_test(self, data):
        return {'is_test': (data['data_df']['Survived'].values == -1)}

    @require('data_df')
    @will_generate('h5py', 'is_valid')
    def gen_is_validation(self, data):
        from sklearn.model_selection import train_test_split
        import numpy as np
        data_df = data['data_df']
        df = pd.DataFrame(False, index=data_df.index, columns=['is_valid'], dtype=bool)
        random_state = np.random.RandomState(1126)
        _, valid_id = train_test_split(
            data_df.loc[(data_df['Survived']!=-1)].index, test_size=0.3,
            random_state=random_state)
        df.loc[valid_id, 'is_valid'] = True
        return df

    @require('data_df')
    @will_generate('h5py', 'label')
    def gen_label(self, data):
        data_df = data['data_df']
        return {'label': data_df['Survived'].values}

    @require('data_df')
    @will_generate('h5py', 'pclass')
    def gen_pclass(self, data):
        from sklearn.preprocessing import OneHotEncoder
        data_df = data['data_df'].copy() # it is mutable
        # set unknown as a class
        data_df.loc[data_df['Pclass'].isnull(), 'Pclass'] = 4
        return {'pclass': OneHotEncoder(sparse=False).fit_transform(
                            data_df['Pclass'].values.reshape((-1, 1)))}

    @require('data_df')
    @will_generate('h5py', 'family_size')
    def gen_family_size(self, data):
        data_df = data['data_df']
        return {'family_size': (data_df['SibSp'] + data_df['Parch']).values}

    @require('data_df')
    @will_generate('h5py', ['age', 'sibsp'])
    def gen_age_sibsp(self, data):
        data_df = data['data_df'].copy()
        # clean up age data
        data_df.loc[data_df['Age'].isnull(), 'Age'] = data_df['Age'].mean()
        return {'age': data_df['Age'].values,
                'sibsp': data_df['SibSp'].values}

def generate_titanic_features(generator, bundle_hdf_path):
    label_list = ['label']
    info_list = ['is_valid', 'is_test']
    id_list = ['passenger_id']
    feature_list = ['family_size', 'sibsp', 'age', 'pclass']

    generator.generate(feature_list + label_list + info_list + id_list)

    bundle_structure = {'label': label_list[0],
                        'info': info_list,
                        'id': id_list[0],
                        'features': feature_list}
    structure_config = {'features': {'concat': True}}
    generator.bundle(bundle_structure, data_bundle_hdf_path=bundle_hdf_path,
            structure_config=structure_config)

def main():
    h5py_hdf_path = os.path.join(
            os.path.dirname(__file__), 'h5py.h5')
    bundle_hdf_path = os.path.join(
            os.path.dirname(__file__), 'data_bundles', 'feature01.h5')
    generator = TitanicFeatureGenerator(h5py_hdf_path,
        os.path.join(os.path.dirname(__file__), 'data', 'train.csv'),
        os.path.join(os.path.dirname(__file__), 'data', 'test.csv'))

    generate_titanic_features(generator, bundle_hdf_path)

if __name__ == '__main__':
    main()
