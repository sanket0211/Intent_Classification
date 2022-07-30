from transformers import AutoModelForSequenceClassification
import config

class Intentmodel():
    def initialize_model(self):
        return AutoModelForSequenceClassification.from_pretrained(config.PRETRAINED_MODEL, num_labels=config.NUM_LABELS)

    def initialize_inference_model(self, model_path):
        return AutoModelForSequenceClassification.from_pretrained(model_path)