import config
from transformers import AutoTokenizer


class Tokenizer():
    def initialize_tokenizer(self):
        return AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL)

    def initialize_inference_model(self, model_path):
        return AutoTokenizer.from_pretrained(model_path)