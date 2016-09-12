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

#pickle_column_names_file = data_path + 'cat_col_names.pkl'
#start_time_column_name = 'L0_S0_D1'
#id_column_name = 'Id'
#dep_var_name = 'Response'

bin_num = 5 ## number of bins to separate data by start_time


## get index grouped by start_time
tmp_train, tmp_test, bins, bin_names = bosch_functions.create_grouped_index_df(bin_num)


'''
processed_data_path = './bosch_complete_processed_data/'

selected_bin_name = '0'
none_selected_window_num = bin_names[:]
none_selected_window_num.append(np.NaN)
none_selected_window_num.remove(selected_bin_name)
skipped_test_row_num = tmp_test.loc[tmp_test['time_window_num'].isin(none_selected_window_num), 'row_num'].tolist()
skipped_train_row_num = tmp_train.loc[tmp_train['time_window_num'].isin(none_selected_window_num), 'row_num'].tolist()

train_data_file = processed_data_path + 'processed_totBins_' + str(len(bin_names)+1) + '_bin_' + selected_bin_name + '_train.csv'
test_data_file = processed_data_path + 'processed_totBins_' + str(len(bin_names)+1) + '_bin_' + selected_bin_name + '_test.csv'
bosch_functions.subset_complete_data_by_index(skipped_train_row_num, skipped_test_row_num, train_data_file, test_data_file)
#'''


#'''
## dict for the yaml file
subset_data_dict = {}

for selected_bin_name, i in zip(bin_names, range(len(bin_names))):
    print 'for bin:', selected_bin_name
    subset_data_dict[selected_bin_name] = {}
    none_selected_window_num = bin_names[:]
    none_selected_window_num.append(np.NaN)
    none_selected_window_num.remove(selected_bin_name)
    subset_data_dict[selected_bin_name]['start_time'] = round(float(bins[i]), 2)
    subset_data_dict[selected_bin_name]['end_time'] = round(float(bins[i+1]), 2)

    skipped_test_row_num = tmp_test.loc[tmp_test['time_window_num'].isin(none_selected_window_num), 'row_num'].tolist()
    skipped_train_row_num = tmp_train.loc[tmp_train['time_window_num'].isin(none_selected_window_num), 'row_num'].tolist()
    print len(skipped_test_row_num), len(skipped_train_row_num), skipped_test_row_num[:10], skipped_train_row_num[:10]
    
    train_data_file = 'processed_totBins_' + str(len(bin_names)+1) + '_bin_' + selected_bin_name + '_train.csv'
    test_data_file = 'processed_totBins_' + str(len(bin_names)+1) + '_bin_' + selected_bin_name + '_test.csv'
    bosch_functions.subset_complete_data_by_index(skipped_train_row_num, skipped_test_row_num, train_data_file, test_data_file)
    subset_data_dict[selected_bin_name]['train_file'] = train_data_file
    subset_data_dict[selected_bin_name]['test_file'] = test_data_file

## for the NaN start_time rows
print 'for bin: NaN', 
none_selected_window_num = bin_names[:]
skipped_test_row_num = tmp_test.loc[tmp_test['time_window_num'].isin(none_selected_window_num), 'row_num'].tolist()
skipped_train_row_num = tmp_train.loc[tmp_train['time_window_num'].isin(none_selected_window_num), 'row_num'].tolist()
selected_bin_name = 'NaN'
subset_data_dict[selected_bin_name] = {}
subset_data_dict[selected_bin_name]['start_time'] = 'NaN'
subset_data_dict[selected_bin_name]['end_time'] = 'NaN'

train_data_file = 'complete_processed_totBins_' + str(len(bin_names)+1) + '_bin_' + selected_bin_name + '_train.csv'
test_data_file  = 'complete_processed_totBins_' + str(len(bin_names)+1) + '_bin_' + selected_bin_name + '_test.csv'
bosch_functions.subset_complete_data_by_index(skipped_train_row_num, skipped_test_row_num, train_data_file, test_data_file)
subset_data_dict[selected_bin_name]['train_file'] = train_data_file
subset_data_dict[selected_bin_name]['test_file'] = test_data_file

## write the information to yml file
print subset_data_dict
with open('complete_subset_data_{}_bins_dict.yml'.format(bin_num), 'w') as outfile:
    yaml.dump(subset_data_dict, outfile, default_flow_style=False)
#'''


