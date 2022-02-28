# Issue-Classifier

Our submission for [NLBSE 2022](https://nlbse2022.github.io/) in the tool competition track.  



##  Tool Paper Abstract
Recent innovations in natural language processing techniques have
led to the development of various tools for assisting software de-
velopers. This paper provides a report of our proposed solution
to the issue report classification task from the NL-Based Software
Engineering workshop. We approach the task of classifying issues
on GitHub repositories using BERT-style models. We propose a neural architecture for the problem that utilizes contextual
embeddings for the text content in the GitHub issues. Besides, we
design additional features for the classification task. We perform a
thorough ablation analysis of the designed features and benchmark
various BERT-style models for generating textual embeddings. Our
proposed solution performs better than the competition organizer’s
method and achieves an 𝐹1 score of 0.8653 (Approx 5% increase). 

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
python src/train.py --DATASET_SUFFIX _dropfeature --MODEL_NAME roberta --EMB_MODEL_CHECKPOINT roberta-base --device gpu
```
Use `--device cpu` if you do not have access to a GPU.

## Predictions

1. Download the trained RoBERTA model from [Google Drive](https://drive.google.com/file/d/1YN70CEIWWidRmqvwPgUGECtVy4NjayIy/view?usp=sharing) and put it in `./data/save/` directory.

2. To generate results on the test data, run:

```predictions
python src/evaluate.py --DATASET_SUFFIX _dropfeature --MODEL_NAME roberta --EMB_MODEL_CHECKPOINT roberta-base --device gpu
```

This assumes that the trained model is present in `/data/save/` directory.
