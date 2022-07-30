import argparse
from statistics import mode
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import evaluate
from utils import Utils
import config
from sklearn.metrics import f1_score
from model import Intentmodel
from tokenizer import Tokenizer
utils_obj = Utils()
model_obj = Intentmodel()
tokenizer_obj = Tokenizer()
intent_index_mapping = config.INTENT_INDEX_MAPPING



index_intent_mapping = utils_obj.get_index_intent_mapping(intent_index_mapping)

def main(args):
    model = model_obj.initialize_inference_model(args.modelfile_path)
    tokenizer = tokenizer_obj.initialize_inference_model(args.modelfile_path)
    test_df = utils_obj.load_data(args.testfile_path)
    transcriptions = test_df["transcription"].tolist()
    actions = test_df["action"].tolist()
    action_index = []
    for act in actions:
        action_index.append(intent_index_mapping[act])

    print("Data Initialized. Prediction begins...")
    predicted_action_index = []
    for text in transcriptions:
        inputs = tokenizer(text, return_tensors="pt")
        logits = model(**inputs).logits
        pred_class_idx = torch.argmax(logits).item()
        predicted_action_index.append(pred_class_idx)
        
    print(f"F1 Score: {f1_score(action_index, predicted_action_index, average='micro')}")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--testfile_path', type=str, required=True)
  parser.add_argument('--modelfile_path', type=str, required=True)
  args = parser.parse_args()
  main(args)