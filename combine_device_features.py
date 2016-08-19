import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data_path = '/home/ymm/data/talkingdata_data/'
gender_age_train_file         = 'gender_age_train.csv'
gender_age_test_file          = 'gender_age_test.csv'
phone_brand_device_model_file = 'phone_brand_device_model.csv'
device_feature_file           = 'devide_feature_data.csv'

phone_brand_device_model = pd.read_csv(data_path + phone_brand_device_model_file)
device_feature_data = pd.read_csv(data_path + device_feature_file, index_col='device_id')

## drop duplicated device_id row
phone_brand_device_model.drop_duplicates('device_id', keep='first', inplace=True)
phone_brand_device_model = phone_brand_device_model.set_index('device_id')
phone_brand_device_model['unique_brand_id'] = phone_brand_device_model.apply(lambda x: x[0] + x[1], axis=1)

def linear_transform_categorical_column(dataFrame, column_name):
    dataFrame[column_name] = LabelEncoder().fit_transform(dataFrame[column_name])
    return

linear_transform_categorical_column(phone_brand_device_model, 'phone_brand')
linear_transform_categorical_column(phone_brand_device_model, 'device_model')
linear_transform_categorical_column(phone_brand_device_model, 'unique_brand_id')

## combine the features with label
gender_age_train = pd.read_csv(data_path + gender_age_train_file, index_col='device_id', usecols=['device_id', 'group'])
train_data = pd.merge(gender_age_train, phone_brand_device_model, how='left', left_index=True)
train_data = pd.merge(train_data, device_feature_data, how='left', left_index=True)

train_file = 'train_data.csv'
train_data.to_csv(data_path + train_file, index=False)


