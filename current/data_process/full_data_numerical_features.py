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
from utils.feature_engineering import NumericalFeatureEngineering

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
feature_type = 'numerical'

start_time = time.time()
## loading the data by using the skipped_row_num list
train_num   = pd.read_csv(join(data_path, train_num_file),   index_col='Id')
test_num    = pd.read_csv(join(data_path, test_num_file),    index_col='Id')

print 'finish loading date using {} seconds'.format(round(time.time() - start_time, 0))

remove_single_value_columns(train_num, 'Response', test=test_num)

#### numerical feature engineering work
start_time = time.time()
train_num_Basics = NumericalFeatureEngineering(train_num)
test_num_Basics = NumericalFeatureEngineering(test_num)

combined_train = pd.concat([train_num, train_num_Basics], axis=1)
combined_test  = pd.concat([test_num, test_num_Basics], axis=1)                                                                            
print 'finish generating numerical features and new dataset using {} minutes'.format(round((time.time()-start_time)/60, 2))
print 'combined train numerical feature shape: {}, combined test numerical features shape: {}'.format(combined_train_num.shape, combined_test_num.shape)


start_time = time.time()
combined_train.to_csv('bosch_train_{}_features.csv'.format(feature_type)) 
combined_test.to_csv('bosch_test_{}_features.csv'.format(feature_type)) 
print 'finish writing {} feature data into csv files using {} minutes'.format(feature_type, round((time.time() - start_time)/60, 2))

