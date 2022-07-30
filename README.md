# Intent_Classification

Here we try to build a classifier to classify intents. You can study more about training and validation data in the Dataset directory.

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python3 train.py --trainfile_path <path to the training file>
```

## Inference with Trained Model

Download the Model files from here: https://drive.google.com/drive/folders/1rf3hhZUPf2ZE0u2KvMNmXo41wUbbFPjf?usp=sharing

```bash
python3 inference.py --testfile_path <path to the test file> --modelfile_path <path to the trained model>
```

