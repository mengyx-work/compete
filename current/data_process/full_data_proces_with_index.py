import os, sys, time
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
train_cat   = pd.read_csv(join(data_path, train_cat_file),   index_col='Id')
test_num    = pd.read_csv(join(data_path, test_num_file),      index_col='Id')
test_dat    = pd.read_csv(join(data_path, test_date_file),     index_col='Id')
test_cat    = pd.read_csv(join(data_path, test_date_file),     index_col='Id')

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
remove_single_value_columns(train_cat, test=test_cat)

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

print 'finish feature engineering staiont features using {} minutes'.format(round((time.time() - start_time)/60, 2))


#### categorical features FE
start_time = time.time()

train_cat['Response'] = train_num['Response']
encode_categorical_by_dep_var(train_cat, test_cat)
train_cat.drop('Response', axis=1, inplace=True)

train_cat_Basics = BasicCat_FeatureEngineering(train_cat)
test_cat_Basics  = BasicCat_FeatureEngineering(train_cat)

combined_train_cat = pd.concat([train_cat, train_cat_Basics], axis=1)
combined_test_cat  = pd.concat([test_cat, test_cat_Basics], axis=1)  
print 'finish generating categorical features using {} seconds'.format(round(time.time() - start_time, 0))


#### numerical feature engineering work
start_time = time.time()
train_num_Basics = NumericalFeatureEngineering(train_num)
test_num_Basics = NumericalFeatureEngineering(test_num)

combined_train_num = pd.concat([train_num, train_num_Basics], axis=1)
combined_test_num  = pd.concat([test_num, test_num_Basics], axis=1)                                                                            
print 'finish generating numercail features and new dataset using {} minutes'.format(round((time.time()-start_time)/60, 2))
print 'combined train numerical feature shape: {}, combined test numerical features shape: {}'.format(combined_train_num.shape, combined_test_num.shape)


#### date features engineering works

## basic features from tmp_train_dat
start_time = time.time()
train_dat_Basics = BasicDate_FeatureEngineering(train_dat)
test_dat_Basics  = BasicDate_FeatureEngineering(test_dat)
print 'finish generating basic date features using {} minutes'.format(round((time.time()-start_time)/60, 2))

## normalized date columns
start_time = time.time()
train_dat_Norm = train_dat.apply(getRelativeTimeColumns, axis=1)
test_dat_Norm  = test_dat.apply(getRelativeTimeColumns, axis=1)
print 'finish generating normalized date features using {} minutes'.format(round((time.time()-start_time)/60, 2))

## remove single-valued columns
remove_single_value_columns(train_dat_Norm, test=test_dat_Norm)

encoder = preprocessing.LabelEncoder()
column_names = train_dat.columns.tolist()
column_names.append('NaN')
encoder.fit(column_names)

## TimeDiff features
start_time = time.time()
train_dat_TimeDiff = train_dat.apply(getTimeChangeColumns, axis=1)
test_dat_TimeDiff  = test_dat.apply(getTimeChangeColumns, axis=1)
TimeDiff_ColumnNames = ['time_diff_start_col', 'time_diff_end_col', 'time_diff_value',
                        'time_ratio_value', 'first_time_value', 'last_time_value', 'first_date_value']
train_dat_TimeDiff.columns = TimeDiff_ColumnNames
test_dat_TimeDiff.columns = TimeDiff_ColumnNames

for column in ['time_diff_start_col', 'time_diff_end_col']:
    train_dat_TimeDiff[column].fillna('NaN', inplace=True)
    train_dat_TimeDiff[column] = encoder.transform(train_dat_TimeDiff[column])
    
    test_dat_TimeDiff[column].fillna('NaN', inplace=True)
    test_dat_TimeDiff[column] = encoder.transform(test_dat_TimeDiff[column])

print 'finish generating date difference features using {} minutes'.format(round((time.time()-start_time)/60, 2))


## section to create timeStep features
start_time = time.time()
unique_value_counts = 6
timeStep_columnNames = []
column_name_columns = []

for i in xrange(unique_value_counts):
    timeStep_columnNames.extend(['time_diff_step_{}'.format(i), 'column_counts_step_{}'.format(i),
                                 'time_cost_step_{}'.format(i), 'first_column_step_{}'.format(i)])
    column_name_columns.append('first_column_step_{}'.format(i))

train_dat_TimeStep = train_dat_Norm.apply(getTimeSteps, axis=1)
test_dat_TimeStep  = test_dat_Norm.apply(getTimeSteps, axis=1)
train_dat_TimeStep.columns = timeStep_columnNames
test_dat_TimeStep.columns  = timeStep_columnNames

for column in column_name_columns:
    train_dat_TimeStep[column].fillna('NaN', inplace=True)
    test_dat_TimeStep[column].fillna('NaN', inplace=True)
    train_dat_TimeStep[column] = encoder.transform(train_dat_TimeStep[column])
    test_dat_TimeStep[column] = encoder.transform(test_dat_TimeStep[column])
print 'finish generating date steps features using {} minutes'.format(round((time.time()-start_time)/60, 2))


## combine all the date features together
combined_train_dat = pd.concat([train_dat_Norm, train_dat_Basics, train_dat_TimeDiff, train_dat_TimeStep], axis=1)
combined_test_dat  = pd.concat([test_dat_Norm, test_dat_Basics, test_dat_TimeDiff, test_dat_TimeStep], axis=1)
print 'combined date shape:', combined_train_dat.shape, combined_test_dat.shape


start_time = time.time()
train_test_datIndex_features = build_IndexFeatures(combined_train_dat, combined_test_dat)
print 'finish index feature generation using {} minutes'.format(round((time.time() - start_time)/60, 2))

start_time = time.time()
combined_train = pd.concat([combined_train_station_dat, combined_train_station_num, combined_train_cat, combined_train_num, combined_train_dat, train_test_datIndex_features.ix[combined_train_num.index, :]], axis=1)
combined_test  = pd.concat([combined_test_station_dat, combined_test_station_num, combined_test_cat,  combined_test_num,  combined_test_dat,  train_test_datIndex_features.ix[combined_test_num.index, :]], axis=1)

combined_train.to_csv('FE_raw_train_full_index.csv') 
combined_test.to_csv('FE_raw_test_full_index.csv') 
print 'finish writing datea into csv files using {} minutes'.format(round((time.time() - start_time)/60, 2))
