import os, sys, time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
import yaml
import cPickle as pickle

sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
import utils.bosch_functions as bosch_functions

data_path = '/home/ymm/bosch/'

train_num_file   = 'train_numeric.csv'
train_cat_file   = 'train_categorical.csv'
train_date_file  = 'train_date.csv'
test_num_file    = 'test_numeric.csv'
test_cat_file    = 'test_categorical.csv'
test_date_file   = 'test_date.csv'

sample_submission_file   = 'sample_submission.csv'



pickle_column_names_file = data_path + 'cat_col_names.pkl'
start_time_column_name = 'L0_S0_D1'
id_column_name = 'Id'
dep_var_name = 'Response'
bin_num = 15 ## number of bins to separate data by start_time


## load the labels and start_time column for train and test data
start_time = time.time()
train_labels = pd.read_csv(data_path + train_num_file, index_col='Id', usecols=['Id', dep_var_name])
train_date_start_columm = pd.read_csv(data_path + train_date_file, index_col='Id', usecols=['Id', start_time_column_name])
test_date_start_columm = pd.read_csv(data_path + test_date_file, index_col='Id', usecols=['Id', start_time_column_name])
end_time = time.time()
print 'data loading takes ', round((end_time - start_time), 1), ' seconds.'

## join the start_time with labels, then drop the NaN in start_time
labeled_start_time = pd.merge(train_labels, train_date_start_columm, how='left', left_index=True, right_index=True)
## this labeled_start_time dataFrame doesn't contain the NaN
## can be directly used for calculating the mquantiles
labeled_start_time = labeled_start_time[~labeled_start_time[start_time_column_name].isnull()]


'''
section to subset the data by start_time
'''
from scipy.stats.mstats import mquantiles

prob_list = [1.*i/bin_num for i in range(1, bin_num)]
quantile_values = mquantiles(labeled_start_time[start_time_column_name], prob=prob_list)

bins = [labeled_start_time[start_time_column_name].min()]
bins.extend(quantile_values)
bins.append(labeled_start_time[start_time_column_name].max())
bin_names = [str(i) for i in range(len(bins)-1)]

## cut the entire dataframe into different time_windows by start_time
tmp_train = train_date_start_columm.copy()
tmp_test = test_date_start_columm.copy()

tmp_train['time_window_num'] = pd.cut(tmp_train[start_time_column_name], bins, labels=bin_names)
tmp_test['time_window_num'] = pd.cut(tmp_test[start_time_column_name], bins, labels=bin_names)
## create a row number column, start index is 1
tmp_train['row_num'] = range(1, (tmp_train.shape[0] + 1))
tmp_test['row_num'] = range(1, (tmp_test.shape[0] + 1))


## for the NaN start_time rows
selected_bin_name = '0'
none_selected_window_num = bin_names[:]
none_selected_window_num.append(np.NaN)
none_selected_window_num.remove(selected_bin_name)
skipped_test_row_num = tmp_test.loc[tmp_test['time_window_num'].isin(none_selected_window_num), 'row_num'].tolist()
skipped_train_row_num = tmp_train.loc[tmp_train['time_window_num'].isin(none_selected_window_num), 'row_num'].tolist()
    
train_data_file = 'processed_totBins_' + str(len(bin_names)+1) + '_bin_' + selected_bin_name + '_train.csv'
test_data_file = 'processed_totBins_' + str(len(bin_names)+1) + '_bin_' + selected_bin_name + '_test.csv'
bosch_functions.load_data_by_index(skipped_train_row_num, skipped_test_row_num, train_data_file, test_data_file)
