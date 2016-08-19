import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack

data_path = '/home/ymm/data/talkingdata_data/'
app_event_file = 'app_events.csv'
processed_app_label_file = 'processed_app_label_table.csv'
sparse_app_labels = pd.read_csv(data_path + processed_app_label_file)
app_event = pd.read_csv(data_path + app_event_file, usecols= ['app_id', 'is_active', 'event_id'])

def update_app_event_column_name(column_names, level):
    corr_column_names = []
    for column_name in column_names:
        corr_column_names.append("active_" + str(level) + "_" + column_name)
    return corr_column_names
    
def split_aggregate_combine(split_column_name, agg_column_name, dataFrame):
    unique_levels = dataFrame[split_column_name].unique()
    combined_dataFrame = None
    for unique_level in unique_levels:
        dataFrame_byLevel = dataFrame.loc[dataFrame[split_column_name] == unique_level, :]
        print "for unique level:", unique_level
        print "data before aggregation:", dataFrame_byLevel.shape
        dataFrame_byLevel = dataFrame_byLevel.drop('app_id', axis = 1).groupby('event_id').aggregate(np.sum)
        ## column name change
        dataFrame_byLevel.columns = update_app_event_column_name(dataFrame_byLevel.columns, unique_level)
        print "data after aggregation:", dataFrame_byLevel.shape
        if combined_dataFrame is None:
            combined_dataFrame = dataFrame_byLevel
        else:
            combined_dataFrame = pd.merge(combined_dataFrame, dataFrame_byLevel, how='outer', left_index=True, right_index=True)    
    return combined_dataFrame

#print "app_event before joining:", app_event.shape
#app_event_labeled = pd.merge(app_event, sparse_app_labels, on='app_id', how='left')
#print "app_event after joining:", app_event_labeled.shape
#agg_app_event_labeled = split_aggregate_combine('is_active', 'event_id', app_event_labeled)

unique_event_ids = app_event['event_id'].unique()
batch_num = 50
agg_app_event_labeled = None

for i in range(batch_num):
    i = i * 1.
    start_index = int(i / batch_num * len(unique_event_ids))
    if i != (batch_num - 1):
        end_index = int((i + 1) / batch_num * len(unique_event_ids))
    else:
        end_index = len(unique_event_ids)

    tmp_app_event = app_event.loc[app_event.event_id.isin(unique_event_ids[start_index:end_index])]
    tmp_app_event_labeled = pd.merge(tmp_app_event, sparse_app_labels, on='app_id', how='left')
    tmp_agg_app_event_labeled = split_aggregate_combine('is_active', 'event_id', tmp_app_event_labeled)
    if agg_app_event_labeled is None:
        agg_app_event_labeled = tmp_agg_app_event_labeled
    else:
        print 'for round:', i, 'the current agg_app_event_labeled shape:', agg_app_event_labeled.shape
        agg_app_event_labeled = pd.concat([agg_app_event_labeled, tmp_agg_app_event_labeled])

print 'agg_app_event_labeled is finished with shape:', agg_app_event_labeled.shape

agg_app_event_labeled_file = 'agg_app_event_labeled_data.csv'
agg_app_event_labeled.to_csv(data_path + agg_app_event_labeled_file)
print 'agg_app_event_labeled is saved into file'


