import os, sys, time, random
import pandas as pd
import numpy as np
from os.path import join

from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold

sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
import utils.bosch_functions as bosch_functions
from utils.wrapped_xgboost import xgboost_classifier
from utils.validation_tools import score_MCC, MCC, create_validation_index
from utils.combined_model import CombinedModel
from utils.data_munge import remove_single_value_columns
from utils.feature_engineering import NumericalFeatureEngineering, getRelativeTimeColumns, BasicDate_FeatureEngineering
from utils.feature_engineering import getTimeChangeColumns, getTimeSteps, build_IndexFeatures, build_sortedData_indexDiff
from utils.feature_engineering import build_sortedData_rowDiff

data_path = '/home/ymm/bosch/'

train_num_file   = 'train_numeric.csv'
train_cat_file   = 'train_categorical.csv'
train_date_file  = 'train_date.csv'
test_num_file    = 'test_numeric.csv'
test_cat_file    = 'test_categorical.csv'
test_date_file   = 'test_date.csv'

sample_submission_file   = 'sample_submission.csv'

start_time_column_name = 'L0_S0_D1'
id_column_name = 'Id'
dep_var_name = 'Response'

start_time = time.time()
data_path = '/home/ymm/kaggle/bosch_data/bosch_processed_data'
raw_data_path = '/home/ymm/bosch/'

train_num_file   = 'train_numeric.csv'
train_cat_file   = 'train_categorical.csv'
train_date_file  = 'train_date.csv'
test_num_file    = 'test_numeric.csv'
test_cat_file    = 'test_categorical.csv'
test_date_file   = 'test_date.csv'

train_file_name = 'selected_combined_train.csv'
test_file_name = 'selected_combined_test.csv'

train = pd.read_csv(join(data_path, train_file_name), index_col='Id')
test = pd.read_csv(join(data_path, test_file_name), index_col='Id')
train_date = pd.read_csv(join(raw_data_path, train_date_file), index_col='Id', usecols=[id_column_name, start_time_column_name])
test_date = pd.read_csv(join(raw_data_path, test_date_file), index_col='Id', usecols=[id_column_name, start_time_column_name])

print 'it takes {} minutes to load train {} and test {}'.format(round((time.time()-start_time)/60, 2), train.shape, test.shape)

single_important_features = ['L3_S30_F3754', 'L3_S30_F3759', 'L3_S33_F3857', 'L3_S33_F3859', 
                            'L3_S30_F3744', 'L3_S30_F3749', 'L3_S30_F3704', 'L3_S33_F3865']

time_important_features = ['time_sum', 'time_mean', 'max_time', 'min_time',
                        'time_diff_value', 'time_ratio_value', 'first_time_value', 'last_time_value']

top_important_features = ['num_L0_mean', 'L3_S30_F3754', 'num_L3_S33_var', 'num_L3_S29_var', 
                        'L3_S30_F3759', 'num_L0_S1_mean', 'num_L0_S0_mean', 'dat_L3_S36_mean',
                        'num_max_min_ratio', 'num_L3_mean', 'dat_L3_var', 'L3_S33_F3857',
                        'dat_L0_mean', 'L3_S33_F3859', 'dat_L3_S33_mean', 'L3_S30_F3744']

combined_important_features = set(top_important_features)
combined_important_features = combined_important_features.union(set(single_important_features))
combined_important_features = combined_important_features.union(set(time_important_features))
combined_important_features = list(combined_important_features)

train_test_date = pd.concat([train_date, test_date], axis=0)
train_test = pd.concat([train[combined_important_features], test[combined_important_features]], axis=0)
train_test['start_time'] = train_test_date[start_time_column_name]
train_test['index'] = train_test.index

new_fea = pd.DataFrame(index = train_test.index)
build_sortedData_indexDiff(train_test, new_fea, combined_important_features, index_col_name='index')

for col_name in combined_important_features:
    build_sortedData_rowDiff(train_test, new_fea, ['start_time'], col_name)
    build_sortedData_rowDiff(train_test, new_fea, ['start_time', 'index'], col_name)
    build_sortedData_rowDiff(train_test, new_fea, ['start_time', col_name, 'index'], 'index')

new_fea_train = new_fea.ix[train.index]
new_fea_test = new_fea.ix[test.index]
print 'the new features, train data shape is {}, test data shape is {}'.format(new_fea_train.shape, new_fea_test.shape)
new_fea_train.to_csv(join(data_path, 'sequence_160_fea_train_fea.csv'))
new_fea_test.to_csv(join(data_path, 'sequence_160_fea_test_fea.csv'))
