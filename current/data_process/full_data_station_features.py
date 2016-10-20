import os, sys, time, gc
import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import preprocessing
sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
import utils.bosch_functions as bosch_functions
from utils.wrapped_xgboost import xgboost_classifier
from utils.validation_tools import score_MCC, MCC, create_validation_index
from utils.models import CombinedModel
from utils.data_munge import remove_single_value_columns
from utils.feature_engineering import NumericalFeatureEngineering, getRelativeTimeColumns, BasicDate_FeatureEngineering
from utils.feature_engineering import getTimeChangeColumns, getTimeSteps, build_IndexFeatures
## station-related feature functions
from utils.feature_engineering import build_column_dict, build_station_features, build_station_index_features 
## categorical fetures related
from utils.feature_engineering import BasicCat_FeatureEngineering, encode_categorical_by_dep_var

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
## loading the data by using the skipped_row_num list
train_num   = pd.read_csv(join(data_path, train_num_file),    index_col='Id')
train_dat   = pd.read_csv(join(data_path, train_date_file),   index_col='Id')
test_num    = pd.read_csv(join(data_path, test_num_file),      index_col='Id')
test_dat    = pd.read_csv(join(data_path, test_date_file),     index_col='Id')

'''
num_rows = 50000
train_num = pd.read_csv(join(data_path, train_num_file),    index_col='Id', nrows=num_rows)
train_dat = pd.read_csv(join(data_path, train_date_file),   index_col='Id', nrows=num_rows)
train_cat = pd.read_csv(join(data_path, train_cat_file),    index_col='Id', nrows=num_rows)
test_num = pd.read_csv(join(data_path, test_num_file),      index_col='Id', nrows=num_rows)
test_dat = pd.read_csv(join(data_path, test_date_file),     index_col='Id', nrows=num_rows)
test_cat = pd.read_csv(join(data_path, test_cat_file),      index_col='Id', nrows=num_rows)
'''

print 'finish loading date using {} seconds'.format(round(time.time() - start_time, 0))

remove_single_value_columns(train_num, 'Response', test=test_num)
remove_single_value_columns(train_dat, test=test_dat)

#### station-based feature engineering works
## create column dictionaries
dat_columns = train_dat.columns.tolist()
num_columns = train_num.columns.tolist()
num_columns.remove(dep_var_name)
dat_col_dict, dat_line_dict = build_column_dict(dat_columns)
num_col_dict, num_line_dict = build_column_dict(num_columns)
dat_col_dict.update(dat_line_dict)
num_col_dict.update(num_line_dict)

start_time = time.time()
train_dat_stations = build_station_features(train_dat, dat_col_dict, 'dat')
test_dat_stations = build_station_features(test_dat, dat_col_dict, 'dat')

train_num_stations = build_station_features(train_num, num_col_dict, 'num')
test_num_stations = build_station_features(test_num, num_col_dict, 'num')

num_station_index = build_station_index_features(train_num_stations, test_num_stations)
dat_station_index = build_station_index_features(train_dat_stations, test_dat_stations)

combined_train_station_num = pd.concat([train_num_stations, num_station_index.ix[train_num_stations.index]], axis=1)
combined_train_station_dat = pd.concat([train_dat_stations, dat_station_index.ix[train_dat_stations.index]], axis=1)

combined_test_station_num = pd.concat([test_num_stations, num_station_index.ix[train_num_stations.index]], axis=1)
combined_test_station_dat = pd.concat([test_dat_stations, dat_station_index.ix[train_dat_stations.index]], axis=1)

combined_train = pd.concat([combined_train_station_dat, combined_train_station_num], axis=1)
combined_test  = pd.concat([combined_test_station_dat,  combined_test_station_num],  axis=1) 
print 'finish feature engineering station-based features using {} minutes'.format(round((time.time() - start_time)/60, 2))

start_time = time.time()
combined_train.to_csv('bosch_train_station_features.csv') 
combined_test.to_csv('bosch_test_station_features.csv') 
print 'finish writing station feature data into csv files using {} minutes'.format(round((time.time() - start_time)/60, 2))
