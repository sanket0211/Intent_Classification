PRETRAINED_MODEL = "distilbert-base-uncased"
MODEL_PATH = "./resultsV1.0/checkpoint-500"
INTENT_INDEX_MAPPING = str_to_int={
    "bring":0,"deactivate":1,"decrease":2,"change language":3,"increase": 4, "activate": 5 
}
MODEL_OUTPUT_PATH = "./results"
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 16
NUM_LABELS = 6
EPOCHS = 5
WEIGHT_DECAY = 0.01
TRAIN_VAL_SPLIT = 0.2