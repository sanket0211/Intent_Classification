from distutils.file_util import copy_file
from sklearn.model_selection import train_test_split
from datasets import Dataset
import config

class Dataloader():
    def prepare_dataset(self, df):
        train_df, test_df = train_test_split(df, test_size=config.TRAIN_VAL_SPLIT)
        train_df.rename(columns = {'transcription':'text', 'action':'label'}, inplace = True)
        test_df.rename(columns = {'transcription':'text', 'action':'label'}, inplace = True)
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        train_dataset = train_dataset.remove_columns(["__index_level_0__"])
        test_dataset = test_dataset.remove_columns(["__index_level_0__"])
        return train_dataset, test_dataset