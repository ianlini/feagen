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


class FeatureConfig(object):

    def __init__(self, feature_list, label_list, extra_data_list,
                 concat_feature_hdf_path):
        self.feature_list = feature_list
        self.label_list = label_list
        self.extra_data_list = extra_data_list
        self.concat_feature_hdf_path = concat_feature_hdf_path


def main():
    feature_generator = LifetimeFeatureGenerator(
        global_feature_hdf_path="global_feature.h5",
        data_csv_path='lifetime.csv')

    feature_config = FeatureConfig(
        label_list=['label'],
        feature_list=['weight', 'height', 'BMI'],
        extra_data_list=[],
        concat_feature_hdf_path="concat_feature.h5")

    feature_generator.generate(feature_config.feature_list
                               + feature_config.extra_data_list
                               + feature_config.label_list)

    fg.save_concat_features(feature_config, "global_feature.h5")


if __name__ == '__main__':
    main()
