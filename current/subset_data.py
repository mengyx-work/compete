import os, sys, time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
import yaml
import cPickle as pickle

data_path = '/home/ymm/bosch/'

train_num_file   = 'train_numeric.csv'
train_cat_file   = 'train_categorical.csv'
train_date_file  = 'train_date.csv'
test_num_file    = 'test_numeric.csv'
test_cat_file    = 'test_categorical.csv'
test_date_file   = 'test_date.csv'

sample_submission_file   = 'sample_submission.csv'


'''
This section loads a fraction of categorical data and save the columns
names into a pickle file named by 'pickle_column_names_file'.
So that the following categorical data loading can directly use explicitly types
'''
def create_categorical_column_name_pickle(train_cat_file, pickle_column_names_file):
    
    tmp_train_cat = pd.read_csv(data_path + train_cat_file, index_col='Id', nrows=1000)

    #for col, dtype in zip(tmp_train_cat.columns, tmp_train_cat.dtypes):
    #    print len(train_cat[col].unique()), dtype

    ## save the column names to pickle file
    col_names = tmp_train_cat.columns.tolist()
    with open(pickle_column_names_file, 'wb') as pickle_file:
        pickle.dump(col_names, pickle_file)


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


def process_date_data(train_date, test_date, start_time_column_name):
    print 'raw date data dimension: ', train_date.shape, test_date.shape
    train_date['start_time'] = train_date[start_time_column_name]
    test_date['start_time'] = test_date[start_time_column_name]
    single_value_column_names = []

    for column in train_date.columns:
        if column != 'start_time':
            train_date[column] = train_date[column] - train_date['start_time']
            test_date[column] = test_date[column] - test_date['start_time']
        if len(train_date[column].unique()) == 1:
            single_value_column_names.append(column)
            
    ## drop single-valued columns        
    train_date.drop(single_value_column_names, axis=1, inplace=True)
    test_date.drop(single_value_column_names, axis=1, inplace=True)
    print 'processed date data dimension: ', train_date.shape, test_date.shape
    

    
def remove_single_value_categorical_columns(train, test):
    print 'raw categorical data dimension: ', train.shape, test.shape
    single_value_column_names = []
    for col in train.columns:
        if len(train[col].unique()) == 1:
            single_value_column_names.append(col)
    
    train.drop(single_value_column_names, axis=1, inplace=True)
    test.drop(single_value_column_names, axis=1, inplace=True)
    print 'processed categorical data dimension: ', train.shape, test.shape



def encode_categorical_data(train, test, fill_missing = False):
    '''
    encoding is an extemely slow process
    So only use the training data to trian the encoder
    '''
    le = LabelEncoder()
    
    if fill_missing:
        train = train.fillna(value='missing')
        test = test.fillna(value='missing')
    
    ## idealy combine the train and test
    #combined = pd.concat([train, test], axis=0)
    counter = 0
    start_time = time.time()
    for col, dtype in zip(train.columns, train.dtypes):
        if dtype == 'object':
            le.fit(pd.concat([train[col], test[col]], axis=0))
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])
                              
        counter += 1
        if counter % 20 == 0:
            print '{} out of {} is process...'.format(str(counter), str(train.shape[1]))
                              
    end_time = time.time()
    print 'encoding process takes ', round((end_time - start_time)), 'seconds'
    
    

def load_data_by_index(skipped_train_row_num, skipped_test_row_num, train_data_file, test_data_file):

    start_time = time.time()

    ## section to load column names for categorical data
    if not os.path.isfile(pickle_column_names_file):
        print 'create new column name pickle file ...'
        create_categorical_column_name_pickle(train_cat_file, pickle_column_names_file)
    
    with open(pickle_column_names_file, 'rb') as pickle_file:
        cat_column_names = pickle.load(pickle_file)
    
    column_types = [np.object] * len(cat_column_names)
    column_types_dict = dict(zip(cat_column_names, column_types))
    ######

    train_date = pd.read_csv(data_path + train_date_file, index_col='Id', skiprows=skipped_train_row_num)
    train_num = pd.read_csv(data_path + train_num_file, index_col='Id', skiprows=skipped_train_row_num)
    train_cat = pd.read_csv(data_path + train_cat_file, index_col='Id', skiprows=skipped_train_row_num, dtype=column_types_dict)

    test_date = pd.read_csv(data_path + test_date_file, index_col='Id', skiprows=skipped_test_row_num)
    test_num = pd.read_csv(data_path + test_num_file, index_col='Id', skiprows=skipped_test_row_num)
    test_cat = pd.read_csv(data_path + test_cat_file, index_col='Id', skiprows=skipped_test_row_num, dtype=column_types_dict)

    end_time = time.time()
    print 'data loading takes ', round((end_time - start_time), 2), 'seconds'
    
    ## process the date data
    process_date_data(train_date, test_date, start_time_column_name)
    print 'finish processing date data ...'
    
    ## process categorical data
    remove_single_value_categorical_columns(train_cat, test_cat)
    encode_categorical_data(train_cat, test_cat, True)
    print 'finish processing categorical data ...'

    ## combine the data and save into csv files
    combined_train = pd.concat([train_cat, train_num, train_date], axis=1)
    combined_test = pd.concat([test_cat, test_num, test_date], axis=1)
    
    combined_train.to_csv(train_data_file)
    combined_test.to_csv(test_data_file)


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
    load_data_by_index(skipped_train_row_num, skipped_test_row_num, train_data_file, test_data_file)
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

train_data_file = 'processed_totBins_' + str(len(bin_names)+1) + '_bin_' + selected_bin_name + '_train.csv'
test_data_file = 'processed_totBins_' + str(len(bin_names)+1) + '_bin_' + selected_bin_name + '_test.csv'
load_data_by_index(skipped_train_row_num, skipped_test_row_num, train_data_file, test_data_file)
subset_data_dict[selected_bin_name]['train_file'] = train_data_file
subset_data_dict[selected_bin_name]['test_file'] = test_data_file

## write the information to yml file
print subset_data_dict
with open('subsest_data_dict.yml', 'w') as outfile:
    yaml.dump(subset_data_dict, outfile, default_flow_style=False)


