import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AdamW

from models import BERTClass
from utils import get_benchmarks, create_modified_dataset


def load_model(args):
    model = BERTClass()
    model.to(args.device)
    output_model = args.SAVED_MODELS_DIR + "BERT_classifier"+ args.DATASET_SUFFIX+".bin"
    load(model,optim,output_model)
    return model


def evaluate_model(model, args):
    create_modified_dataset(args, dtype = ['test'])
    test_df = pd.read_csv(args.DATASET_DIR + args.EMB_MODEL_CHECKPOINT + "_test" + args.DATASET_SUFFIX + ".csv")
    test_dataset = CustomTextDataset(test_df)

    test_loader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
    optim = AdamW(model.parameters(), lr=args.LEARNING_RATE)

    model.eval()
    print("[INFO] Model loaded!")

    # Evaluating the model
    y_true = []
    y_pred = []
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        features = batch["features"].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask, token_type_ids,features)
        y_true += labels.cpu().detach().numpy().tolist()
        y_pred += torch.argmax(outputs, dim=1).cpu().detach().numpy().tolist()

    return get_benchmarks(y_true,y_pred)


if __name__ == '__main__':
    args, logging_args = get_arguments()
    set_random_seed(args.seed)
    model = load_model(args)
    _ = evaluate_model(model,args)