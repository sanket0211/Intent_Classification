import argparse
from ast import arg
from datasets import Dataset, load_dataset
from utils import Utils
import config
from tokenizer import Tokenizer
import pandas as pd
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from model import Intentmodel
from dataloader import Dataloader
utils_obj = Utils()
data_obj = Dataloader()
model_obj = Intentmodel()
tokenizer_obj = Tokenizer()
intent_index_mapping = config.INTENT_INDEX_MAPPING

tokenizer = tokenizer_obj.initialize_tokenizer()
model = model_obj.initialize_model()

def preprocess_function(examples):
    tokenized_batch = tokenizer(examples['text'], truncation=True)
    tokenized_batch["label"] = [intent_index_mapping[label] for label in examples["label"]]
    return tokenized_batch

def main(args):
    df = utils_obj.load_data(args.trainfile_path)
    df.drop(["path", "object", "location"], axis=1, inplace=True)
    train_dataset, test_dataset = data_obj.prepare_dataset(df)
    tokenized_train_data = train_dataset.map(preprocess_function, batched=True)
    tokenized_test_data = test_dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    
    training_args = TrainingArguments(
        output_dir=config.MODEL_OUTPUT_PATH,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.VAL_BATCH_SIZE,
        num_train_epochs=config.EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_test_data,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    trainer.train()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--trainfile_path', type=str, required=True)
  args = parser.parse_args()
  main(args)

