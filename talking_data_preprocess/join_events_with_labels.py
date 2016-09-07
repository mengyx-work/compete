import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack

data_path = '/home/ymm/data/talkingdata_data/'
event_file = 'events.csv'
agg_app_event_labeled_file = 'agg_app_event_labeled_data.csv'
agg_app_event_labeled = pd.read_csv(data_path + agg_app_event_labeled_file, index_col='event_id')
print 'agg_app_event_labeled is loaded from file with shape:', agg_app_event_labeled.shape

events = pd.read_csv(data_path + event_file, index_col='event_id')
events_labeled = pd.merge(events, agg_app_event_labeled, how='left', left_index=True, right_index=True)
print 'the events shape:', events_labeled.shape
print '#unique device_id:', len(events_labeled.device_id.unique())
print '#unique event_id:', len(events_labeled.index.unique())
events_labeled = pd.merge(events, agg_app_event_labeled, how='left', left_index=True, right_index=True)

## save events_labeled to file
events_labeled_file = 'events_labeled_data.csv'
events_labeled.to_csv(data_path + events_labeled_file)
print 'events_labeled is saved into file:', events_labeled_file

