import pandas as pd
from sklearn.model_selection import train_test_split
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

    # @features(skip_if_exist=True)
    # @require_intermediate_data('data_df')
    # @will_generate_one_of(r"\w+_divided_by_\w+")
    # def gen_divided_by(self, will_generate_key, data):
    #     data_df = data['data_df']
    #     division_result = data_df[data1] / data_df[data2]
    #     return {will_generate_key: division_result}

    @features(skip_if_exist=True)
    @require_intermediate_data('data_df')
    @will_generate('is_in_test_set')
    def gen_is_in_test_set(self, data):
        data_df = data['data_df']
        _, test_id = train_test_split(
            data_df.index, test_size=0.5, random_state=0)
        is_in_test_set = data_df.index.isin(test_id)
        return {'is_in_test_set': is_in_test_set}

# import ipdb; ipdb.set_trace()

def main():
    feature_generator = LifetimeFeatureGenerator(
        global_feature_hdf_path="global_feature.h5",
        data_csv_path='lifetime.csv')

    feature_list = ['weight', 'height', 'BMI']
    # feature_list = ['weight', 'height', 'BMI', 'weight_divided_by_height']
    label_list = ['label']
    test_filter_list = ['is_in_test_set']

    feature_generator.generate(feature_list + label_list + test_filter_list)

    fg.save_concat_features(
        feature_list=feature_list,
        global_feature_hdf_path="global_feature.h5",
        concat_feature_hdf_path="concat_feature.h5",
        extra_data={'label': label_list, 'test_filter_list': test_filter_list})


if __name__ == '__main__':
    main()
