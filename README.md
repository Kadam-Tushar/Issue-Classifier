# Issue-Classifier

## Setup

1. Install requirements with
```setup
conda env create -f environment.yml
```
2. Download data using the Bash script `data/get_data.sh`
```
cd data && ./get_data.sh
```

## Training

To train the model in the paper, run this command:

```train
python src/train.py --DATASET_SUFFIX _roberta --MODEL_NAME roberta --EMB_MODEL_CHECKPOINT roberta-base --device gpu
```
Use `--device cpu` if you do not have access to a GPU.

## Predictions

1. Download the trained RoBERTA model from LINK[TODO] and put it in `./data/save/` directory.

2. To generate results on the test data, run:

```predictions
python src/evaluate.py --DATASET_SUFFIX _roberta --MODEL_NAME roberta --EMB_MODEL_CHECKPOINT roberta-base --device gpu
```

This assumes that the trained model is present in `/data/save/` directory.
