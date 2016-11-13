import pandas as pd
import feagen as fg
from feagen.decorators import (
    will_generate,
    require_intermediate_data,
    features,
    intermediate_data,
)


class LifetimeFeatureGenerator(fg.FeatureGenerator):

    def __init__(self, global_feature_hdf_path, data_csv_path):
        super(LifetimeFeatureGenerator, self).__init__(global_feature_hdf_path)
        self.data_csv_path = data_csv_path

    @intermediate_data(skip_if_exist=True, show_skip=False)
    @will_generate('data_df')
    def gen_data_df(self):
        return {'data_df': pd.read_csv(self.data_csv_path, index_col='id')}

    @features(skip_if_exist=True)
    @require_intermediate_data('data_df')
    @will_generate('label')
    def gen_label(self, data):
        data_df = data['data_df']
        return {'label': data_df['lifetime']}

    @features(skip_if_exist=True)
    @require_intermediate_data('data_df')
    @will_generate(['weight', 'height'])
    def gen_raw_data_features(self, data):
        data_df = data['data_df']
        return data_df[['weight', 'height']]

    @features(skip_if_exist=True)
    @require_intermediate_data('data_df')
    @will_generate('BMI')
    def gen_bmi(self, data):
        data_df = data['data_df']
        bmi = data_df['weight'] / ((data_df['height'] / 100) ** 2)
        return {'BMI': bmi}


def main():
    feature_generator = LifetimeFeatureGenerator(
        global_feature_hdf_path="global_feature.h5",
        data_csv_path='lifetime.csv')

    feature_list = ['weight', 'height', 'BMI']
    label_list = ['label']

    feature_generator.generate(feature_list + label_list)

    fg.save_concat_features(
        feature_list=feature_list,
        global_feature_hdf_path="global_feature.h5",
        concat_feature_hdf_path="concat_feature.h5",
        extra_data={'label': label_list})


if __name__ == '__main__':
    main()
