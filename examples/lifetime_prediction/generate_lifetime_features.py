import pandas as pd
import feagen as fg
from feagen.decorators import will_generate

from registered_decorators import (
    require_intermediate_data,
    features,
    intermediate_data,
    FEATURE_FUNC_DICT,
)


class LifetimeFeatureGenerator(fg.FeatureGenerator):

    def __init__(self, global_feature_hdf_path, data_df):
        super(LifetimeFeatureGenerator, self).__init__(global_feature_hdf_path,
                                                       FEATURE_FUNC_DICT)
        self.data_df = data_df

    @features(skip_if_exist=True)
    @will_generate('label')
    def gen_label(self):
        return {'label': self.data_df['lifetime']}

    @features(skip_if_exist=True)
    @will_generate(['weight', 'height'])
    def gen_raw_data_features(self):
        return self.data_df[['weight', 'height']]

    @features(skip_if_exist=True)
    @will_generate('BMI')
    def gen_bmi(self):
        bmi = self.data_df['weight'] / ((self.data_df['height']/100) ** 2)
        return {'BMI': bmi}


class FeatureConfig(object):

    def __init__(self, feature_list, label_list, extra_data_list,
                 concat_feature_hdf_path):
        self.feature_list = feature_list
        self.label_list = label_list
        self.extra_data_list = extra_data_list
        self.concat_feature_hdf_path = concat_feature_hdf_path


def main():
    data_df = pd.read_csv('lifetime.csv', index_col='id')

    feature_generator = LifetimeFeatureGenerator(
        global_feature_hdf_path="global_feature.h5",
        data_df=data_df)

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
