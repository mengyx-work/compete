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



def getCat_LevelFeatures(series):
    feature_array = []
    tmp_series = series.dropna()
    for level in sorted_level_list:
        if sum(tmp_series.isin([level])):
            level_index = tmp_series[tmp_series == level].index
            feature_array.extend([level_index[0], level_index[-1], len(level_index)])
        else:
            feature_array.extend(['NaN', 'NaN', 0])
    return pd.Series(feature_array)       
    #print series.value_counts()       
    value_counts = series.value_counts()



def BasicDate_FeatureEngineering(tmp_train_dat):
    ## feature engineering on the date features
    encoder = preprocessing.LabelEncoder()
    column_names = tmp_train_dat.columns.tolist()
    column_names.append('NaN')
    encoder.fit(column_names)
    dat_new_fea = pd.DataFrame()
    
    if 'L0_S0_D1' in tmp_train_dat.columns:
        dat_new_fea['start_time'] = tmp_train_dat['L0_S0_D1']
        
    dat_new_fea['time_sum'] = tmp_train_dat.sum(axis=1)
    dat_new_fea['time_mean'] = tmp_train_dat.mean(axis=1)
    dat_new_fea['dat_nan_count'] = tmp_train_dat.isnull().sum(axis=1)
    dat_new_fea['max_time'] = tmp_train_dat.max(axis=1)
    dat_new_fea['min_time'] = tmp_train_dat.min(axis=1)
    dat_new_fea['dat_max_min_diff'] = dat_new_fea['max_time'] - dat_new_fea['min_time']
    dat_new_fea['dat_max_min_ratio'] = dat_new_fea['min_time'] / dat_new_fea['max_time']

    dat_new_fea['dat_idxmax'] = tmp_train_dat.idxmax(axis=1)
    dat_new_fea['dat_idxmax'].fillna('NaN', inplace=True)
    dat_new_fea['dat_idxmax'] = encoder.transform(dat_new_fea['dat_idxmax'])
    dat_new_fea['dat_idxmin'] = tmp_train_dat.idxmin(axis=1)
    dat_new_fea['dat_idxmin'].fillna('NaN', inplace=True)
    dat_new_fea['dat_idxmin'] = encoder.transform(dat_new_fea['dat_idxmin'])
    return dat_new_fea



def NumericalFeatureEngineering(df, col_ignore = ['Response']):
    tmp_df = df.loc[:, ~df.columns.isin(col_ignore)]
    new_fea_df = pd.DataFrame()
    encoder = preprocessing.LabelEncoder()
    column_names = tmp_df.columns.tolist()
    column_names.append('NaN')
    encoder.fit(column_names)
    
    new_fea_df['num_mean'] = tmp_df.mean(axis=1)
    
    new_fea_df['num_sum'] = tmp_df.sum(axis=1)
    new_fea_df['num_max'] = tmp_df.max(axis=1)
    new_fea_df['num_min'] = tmp_df.min(axis=1)
    new_fea_df['num_max_min_ratio'] = new_fea_df['num_min'] / new_fea_df['num_max']
    new_fea_df['num_max_min_ratio'] = new_fea_df['num_max_min_ratio'].replace([np.inf, -np.inf], np.NaN)
    new_fea_df['num_nan_col_count'] = tmp_df.isnull().sum(axis=1)
    new_fea_df['num_reg_col_count'] = tmp_df.shape[1] - tmp_df.isnull().sum(axis=1)
    new_fea_df['num_idxmax'] = tmp_df.idxmax(axis=1)
    new_fea_df['num_idxmax'].fillna('NaN', inplace=True)
    new_fea_df['num_idxmax'] = encoder.transform(new_fea_df['num_idxmax'])
    new_fea_df['num_idxmin'] = tmp_df.idxmin(axis=1)
    new_fea_df['num_idxmin'].fillna('NaN', inplace=True)
    new_fea_df['num_idxmin'] = encoder.transform(new_fea_df['num_idxmin'])
    return new_fea_df



def getRelativeTimeColumns(series):
    '''
    normalize the time features by
    the start_time, the first none-NaN
    value
    '''
    if series[0] == np.NaN:
        start_time = series.dropna().index[0]
    else:
        start_time = series[0]
    new_series = series - start_time
    return new_series
   


def getTimeSteps(series, unique_value_counts = 10):
    '''
    in each row/series, use the sorted value_count
    to find the time steps and use the value, counts
    and column_index as features
    '''
    value_counts = series.value_counts()
    value_counts.sort_index(inplace=True)

    if 0. in value_counts.index:
        value_counts = value_counts[value_counts.index != 0.]
        
    available_counts = value_counts.shape[0]
    feature_array = []
    for i in xrange(unique_value_counts):
        if i < available_counts:
            date_value = value_counts.index[i]
            counts = value_counts[date_value]
            first_index = series[series == date_value].index[0]
            avg_time_cost = date_value / counts
            feature = [date_value, counts, avg_time_cost, first_index]
        else:
            feature = [np.NaN, 0, 0, 'NaN']
        feature_array.extend(feature)

    return pd.Series(feature_array)



def getTimeChangeColumns(series):
    start_time = series[0]
    tmp_series = series.dropna()
    if start_time == np.NaN:
        first_index     = tmp_series.index[0]
        last_index      = tmp_series.index[-1]
        first_id_value  = tmp_series[first_index]
        last_id_value   = tmp_series[last_index]
        first_num_value = first_id_value
        time_diff       = last_id_value - first_id_value
        time_ratio      = last_id_value / first_id_value
        return pd.Series([first_index, last_index, time_diff, time_ratio, 
                          first_id_value, last_id_value, first_num_value])
    else:
        first_num_value = start_time
        if np.sum(tmp_series != start_time) == 0:
            return pd.Series(['NaN', 'NaN', np.NaN, np.NaN, np.NaN, np.NaN, first_num_value])
        else:
            first_index     = tmp_series.index[tmp_series != start_time][0]
            last_index      = tmp_series.index[tmp_series != start_time][-1]
            first_id_value  = series[first_index]
            last_id_value   = series[last_index]
            first_id_value  = tmp_series[first_index]
            last_id_value   = tmp_series[last_index]
            time_diff       = last_id_value - first_id_value
            time_ratio      = last_id_value / first_id_value

            return pd.Series([first_index, last_index, time_diff, time_ratio,
                              first_id_value, last_id_value, first_num_value])


    

def build_IndexFeatures(combined_train_dat):
    dat_new_fea = pd.DataFrame()
    dat_new_fea['first_time_index']  = combined_train_dat['first_time_value'].argsort() + 1
    dat_new_fea['last_time_index']   = combined_train_dat['last_time_value'].argsort() + 1
    dat_new_fea['index_ratio']       = dat_new_fea['first_time_index'] / dat_new_fea['last_time_index']
    dat_new_fea['index']             = combined_train_dat.index

    if 'start_time' in combined_train_dat.columns:
        dat_new_fea['start_time_diff']          = combined_train_dat['start_time'].diff()
        dat_new_fea['start_time_index']         = combined_train_dat['start_time'].argsort() + 1
        dat_new_fea['start_time_index_ratio_1'] = dat_new_fea['first_time_index'] / dat_new_fea['index']
        dat_new_fea['start_time_index_ratio_2'] = dat_new_fea['last_time_index'] / dat_new_fea['index']
    
    dat_new_fea['time_ratio_value_index']    = combined_train_dat['time_ratio_value'].argsort() + 1
    dat_new_fea['first_time_value_index']    = combined_train_dat['first_time_value'].argsort() + 1
    dat_new_fea['first_date_value_index']    = combined_train_dat['first_date_value'].argsort() + 1
    dat_new_fea['first_date_value_index_ratio_1'] = dat_new_fea['first_time_index'] / dat_new_fea['index']
    dat_new_fea['first_date_value_index_ratio_2'] = dat_new_fea['last_time_index'] / dat_new_fea['index']

    return dat_new_fea



### section to prepare the raw data, whether separate them by the start_time

bin_num = 1 ## number of bins to separate data by start_time
tmp_train, tmp_test, bins, bin_names = bosch_functions.create_grouped_index_df(bin_num)

def creat_non_selected_window_num(bin_num, select_bin = '0'):
    none_selected_window_num = [np.NaN]
    for i in range(bin_num):
        if str(i) != select_bin:
            none_selected_window_num.append(str(i))
    return none_selected_window_num
    
none_selected_window_num = creat_non_selected_window_num(bin_num, '0')
## select NaN data only
#none_selected_window_num = ['0']

skipped_test_row_num = tmp_test.loc[tmp_test['time_window_num'].isin(none_selected_window_num), 'row_num'].tolist()
skipped_train_row_num = tmp_train.loc[tmp_train['time_window_num'].isin(none_selected_window_num), 'row_num'].tolist()

start_time = time.time()
## loading the data by using the skipped_row_num list
train_num = pd.read_csv(join(data_path, train_num_file),  skiprows=skipped_train_row_num,  index_col='Id')
train_dat = pd.read_csv(join(data_path, train_date_file), skiprows=skipped_train_row_num,  index_col='Id')
train_cat = pd.read_csv(join(data_path, train_cat_file),  skiprows=skipped_train_row_num,  index_col='Id')
test_num = pd.read_csv(join(data_path, test_num_file),  skiprows=skipped_test_row_num,  index_col='Id')
test_dat = pd.read_csv(join(data_path, test_date_file), skiprows=skipped_test_row_num,  index_col='Id')
test_cat = pd.read_csv(join(data_path, test_cat_file),  skiprows=skipped_test_row_num,  index_col='Id')

print 'finish loading date using {} seconds'.format(round(time.time() - start_time, 0))
print 'dat shape: {}, num shape: {}, cat shape: {}'.format(train_dat.shape, train_num.shape, train_cat.shape)


#tmp_train_num = train_num
#tmp_train_dat = train_dat
#tmp_train_cat = train_cat

encoder = preprocessing.LabelEncoder()
cat_column_names = train_cat.columns.tolist()
cat_column_names.append('NaN')
encoder.fit(cat_column_names)

remove_single_value_columns(train_cat)
remove_single_value_columns(train_num)
remove_single_value_columns(train_dat)

## create new categorical features

unique_value_set = set()
for col in train_cat.columns:
    uniques = train_cat[col].dropna().unique()
    unique_value_set |= set(uniques)
    
level_mapping_dict = {}
for i, value in enumerate(sorted(list(unique_value_set))):
    level_mapping_dict[value] = i + 1

sorted_level_list = list(unique_value_set)
sorted_level_list = sorted(sorted_level_list)

start_time = time.time()
train_levelFeatures = train_cat.apply(getCat_LevelFeatures, axis=1)
test_levelFeatures  = test_cat.apply(getCat_LevelFeatures, axis=1)
print 'finish crete new categorical feature using {} minutes'.format(round((time.time() - start_time)/60., 2))

column_name_features = []
cat_fea_names = []
for level in sorted_level_list:
    cat_fea_names.extend(['level_{}_start_column'.format(level), 
                          'level_{}_end_column'.format(level), 
                          'level_{}_column_counts'.format(level)])
    
    column_name_features.extend(['level_{}_start_column'.format(level), 
                                 'level_{}_end_column'.format(level)])

## assign new names to features
train_levelFeatures.columns = cat_fea_names
test_levelFeatures.columns = cat_fea_names

for col in column_name_features:
    train_levelFeatures[col] = encoder.transform(train_levelFeatures[col]) 
    test_levelFeatures[col]  = encoder.transform(test_levelFeatures[col]) 

combined_train_cat = pd.concat([train_cat, train_levelFeatures], axis=1)
combined_test_cat  = pd.concat([test_cat, test_levelFeatures], axis=1)
print 'combined train categorical feature shape: {}, combined test categorical features shape: {}'.format(combined_train_cat.shape, combined_test_cat.shape)



#### numerical feature engineering work
train_num_Basics = NumericalFeatureEngineering(train_num)
test_num_Basics = NumericalFeatureEngineering(test_num)

combined_train_num = pd.concat([train_num, train_num_Basics], axis=1)
combined_test_num  = pd.concat([test_num, test_num_Basics], axis=1)
print 'combined train numerical feature shape: {}, combined test numerical features shape: {}'.format(combined_train_num.shape, combined_test_num.shape)



start_time = time.time()
## normalized date columns
train_dat_Norm = train_dat.apply(getRelativeTimeColumns, axis=1)
test_dat_Norm  = test_dat.apply(getRelativeTimeColumns, axis=1)
## basic features from tmp_train_dat
train_dat_Basics = BasicDate_FeatureEngineering(train_dat)
test_dat_Basics  = BasicDate_FeatureEngineering(test_dat)

encoder = preprocessing.LabelEncoder()
column_names = train_dat.columns.tolist()
column_names.append('NaN')
encoder.fit(column_names)
    
train_dat_TimeDiff = train_dat.apply(getTimeChangeColumns, axis=1)
test_dat_TimeDiff  = train_dat.apply(getTimeChangeColumns, axis=1)
TimeDiff_ColumnNames = ['time_diff_start_col', 'time_diff_end_col', 'time_diff_value', 
                        'time_ratio_value', 'first_time_value', 'last_time_value', 'first_date_value']
train_dat_TimeDiff.columns = TimeDiff_ColumnNames                   
test_dat_TimeDiff.columns = TimeDiff_ColumnNames

for column in ['time_diff_start_col', 'time_diff_end_col']:
    train_dat_TimeDiff[column].fillna('NaN', inplace=True)
    train_dat_TimeDiff[column] = encoder.transform(train_dat_TimeDiff[column])  

    test_dat_TimeDiff[column].fillna('NaN', inplace=True)
    test_dat_TimeDiff[column] = encoder.transform(test_dat_TimeDiff[column])  
   


## section to create timeStep features
unique_value_counts = 10
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

combined_train_dat = pd.concat([train_dat_Norm, train_dat_Basics, train_dat_TimeDiff, train_dat_TimeStep], axis=1)
combined_test_dat  = pd.concat([test_dat_Norm, test_dat_Basics, test_dat_TimeDiff, test_dat_TimeStep], axis=1)
print 'finish feature engineering date using {} minutes'.format(round((time.time() - start_time)/60, 2))

remove_single_value_columns(combined_train_num)
remove_single_value_columns(combined_train_dat)

train_datIndex_features = build_IndexFeatures(combined_train_dat)
test_datIndex_features  = build_IndexFeatures(combined_test_dat)

combined_train = pd.concat([combined_train_num, combined_train_dat, combined_train_cat, train_datIndex_features], axis=1)
combined_train.to_csv('bosch_FE_raw_train_data_regular.csv')
combined_test = pd.concat([combined_test_num, combined_test_dat, combined_test_cat, test_datIndex_features], axis=1)
combined_test.to_csv('bosch_FE_raw_test_data_regular.csv')
