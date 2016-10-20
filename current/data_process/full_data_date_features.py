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
from utils.feature_engineering import getRelativeTimeColumns, BasicDate_FeatureEngineering
from utils.feature_engineering import getTimeChangeColumns, getTimeSteps

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
feature_type = 'date'

start_time = time.time()
train_dat   = pd.read_csv(join(data_path, train_date_file),   index_col='Id')
test_dat    = pd.read_csv(join(data_path, test_date_file),    index_col='Id')
print 'finish loading date using {} seconds'.format(round(time.time() - start_time, 0))

remove_single_value_columns(train_dat, test = test_dat)

#### date features engineering works ####

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
combined_train = pd.concat([train_dat_Norm, train_dat_Basics, train_dat_TimeDiff, train_dat_TimeStep], axis=1)
combined_test  = pd.concat([test_dat_Norm, test_dat_Basics, test_dat_TimeDiff, test_dat_TimeStep], axis=1)
print 'combined date shape:', combined_train.shape, combined_test.shape



start_time = time.time()
combined_train.to_csv('bosch_train_{}_features.csv'.format(feature_type)) 
combined_test.to_csv('bosch_test_{}_features.csv'.format(feature_type)) 
print 'finish writing {} feature data into csv files using {} minutes'.format(feature_type, round((time.time() - start_time)/60, 2))

