import os, sys, time
import pandas as pd
import numpy as np
from os.path import join

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


start_time_column_name = 'L0_S0_D1'
id_column_name = 'Id'
dep_var_name = 'Response'

nrows = 10000
bin_num = 1 ## number of bins to separate data by start_time
tmp_train, tmp_test, bins, bin_names = bosch_functions.create_grouped_index_df(bin_num)

## create the skipped row numbers
none_selected_window_num = bin_names[:]
skipped_test_row_num = tmp_test.loc[tmp_test['time_window_num'].isin(none_selected_window_num), 'row_num'].tolist()
skipped_train_row_num = tmp_train.loc[tmp_train['time_window_num'].isin(none_selected_window_num), 'row_num'].tolist()

train_cat_cols  = pd.read_csv(join(data_path, train_cat_file), index_col=id_column_name, nrows=0)
train_date_cols = pd.read_csv(join(data_path, train_date_file), index_col=id_column_name, nrows=0)
train_num_cols  = pd.read_csv(join(data_path, train_num_file), index_col=id_column_name, nrows=0)

### section to obtain the feature impotance from xgboost model
bin_0_data_path = '/home/ymm/kaggle/bosch/data_2_bins_xgb_combined_models/data_bin_0_models'
bin_1_data_path = '/home/ymm/kaggle/bosch/data_2_bins_xgb_combined_models/data_bin_1_models'

def collect_feature_names(data_path, fea_name='feature', thres_name = None, thres = 10):
    csv_files = [f for f in os.listdir(data_path) if '.csv' in f]
    feature_names = set()
    for file_name in csv_files:
        data = pd.read_csv(join(data_path, file_name), index_col=0)
        if thres_name is None:
            feature_names = feature_names.union(data[fea_name])
        else:
            feature_names = feature_names.union(data.loc[data[thres_name] > thres, fea_name])
            
    return feature_names

## collect feature names based on the fscore
bin_0_selected_col_names = collect_feature_names(bin_0_data_path, 'feature', 'fscore', 10)
bin_1_selected_col_names = collect_feature_names(bin_1_data_path, 'feature', 'fscore', 10)
bin_regular_selected_col_name = bin_0_selected_col_names.union(bin_1_selected_col_names) 

selected_cat_col_names =  train_cat_cols.columns[train_cat_cols.columns.isin(bin_regular_selected_col_name)].tolist()
selected_num_col_names =  train_num_cols.columns[train_num_cols.columns.isin(bin_regular_selected_col_name)].tolist()

selected_cat_col_names.extend([id_column_name])
selected_num_col_names.extend([id_column_name, dep_var_name])

start_time = time.time()
train_cat  = pd.read_csv(join(data_path, train_cat_file),  index_col='Id', skiprows=skipped_train_row_num, usecols=selected_cat_col_names)
test_cat   = pd.read_csv(join(data_path, test_cat_file),   index_col='Id', skiprows=skipped_train_row_num, usecols=selected_cat_col_names)
train_num  = pd.read_csv(join(data_path, train_num_file),  index_col='Id', skiprows=skipped_train_row_num, usecols=selected_num_col_names)
test_num   = pd.read_csv(join(data_path, test_num_file),   index_col='Id', skiprows=skipped_train_row_num, usecols=selected_num_col_names.remove(dep_var_name))
print 'finish reading data by columns selected using xgboost feature importance, using {} seconds.'.format(round(time.time() - start_time, 2))

start_time = time.time()
combined_cat = pd.concat([train_cat, test_cat])
## convert to string so that column is categorical
combined_cat = combined_cat.astype(str)
## One-Hot encode all the categorical columns
oneHot_combined_cat = pd.get_dummies(combined_cat, dummy_na=True)

train_index = train_cat.index
test_index  = test_cat.index
oneHot_train_cat = oneHot_combined_cat.ix[train_index]
oneHot_test_cat  = oneHot_combined_cat.ix[test_index]
print 'finish OneHot encoding the categorical columns, using {} seconds'.format(round(time.time() - start_time, 2))

start_time = time.time()
combined_train = pd.merge(oneHot_train_cat, train_num, how='outer', left_index=True, right_index=True)
combined_test  = pd.merge(oneHot_test_cat,  test_num,  how='outer', left_index=True, right_index=True)
print 'finish merging dataFrame using {} seconds'.format(round(time.time() - start_time, 2))

start_time = time.time()
combined_train.to_csv('bosch_processed_selected_regular_train_data.csv')
combined_test.to_csv('bosch_processed_selected_regular_test_data.csv')
print 'finish saving .csv files using {} seconds'.format(round(time.time() - start_time, 2))
print 'finish generating the processed and selected regualr data...'
