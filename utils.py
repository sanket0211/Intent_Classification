import pandas as pd
import random
from sklearn.model_selection import train_test_split
from datasets import Dataset
class Utils:
    """
    Model Utility Methods
    """
    
    
    def get_index_intent_mapping(self, intent_index_mapping):
        index_intent_mapping={}
        for key in intent_index_mapping:
            index_intent_mapping[intent_index_mapping[key]]=key
        return index_intent_mapping
    
    def load_data(self, file):
        df = pd.read_csv(file,index_col=False)
        return df
    
    def get_data_stats(self, df):
        transcriptions = df['transcription']
        actions = df['action']
        objects = df['object']
        locations = df['location']
        return len(transcriptions), len(list(set(actions))), len(list(set(objects))), len(list(set(locations)))
    
    def get_action_freq(self, df):
        actions = df['action']
        action_freq_dict={}
        for act in actions:
            if act not in action_freq_dict:
                action_freq_dict[act]=0
            action_freq_dict[act]+=1
        return action_freq_dict
  
    def get_object_freq(self, df):
        obj = df['object']
        obj_freq_dict={}
        for o in obj:
            if o not in obj_freq_dict:
                obj_freq_dict[o]=0
            obj_freq_dict[o]+=1
        return obj_freq_dict

    def get_location_freq(self, df):
        location = df['location']
        location_freq_dict={}
        for loc in location:
            if loc not in location_freq_dict:
                location_freq_dict[loc]=0
            location_freq_dict[loc]+=1
        return location_freq_dict

    