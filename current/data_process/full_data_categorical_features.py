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
train_cat   = pd.read_csv(join(data_path, train_cat_file),   index_col='Id')
train_num   = pd.read_csv(join(data_path, train_num_file),   index_col='Id', usecols=['Id', 'Response'])
test_cat    = pd.read_csv(join(data_path, test_cat_file),    index_col='Id')

print 'finish loading date using {} seconds'.format(round(time.time() - start_time, 0))

remove_single_value_columns(train_cat, test = test_cat)

#### categorical features FE
start_time = time.time()

train_cat['Response'] = train_num['Response']
encode_categorical_by_dep_var(train_cat, test_cat)
train_cat.drop('Response', axis=1, inplace=True)

train_cat_Basics = BasicCat_FeatureEngineering(train_cat)
test_cat_Basics  = BasicCat_FeatureEngineering(train_cat)

combined_train = pd.concat([train_cat, train_cat_Basics], axis=1)
combined_test  = pd.concat([test_cat, test_cat_Basics], axis=1)  
print 'finish generating categorical features using {} seconds'.format(round(time.time() - start_time, 0))

start_time = time.time()
combined_train.to_csv('bosch_train_categorical_features.csv') 
combined_test.to_csv('bosch_test_categorical_features.csv') 
print 'finish writing categorical feature data into csv files using {} minutes'.format(round((time.time() - start_time)/60, 2))
