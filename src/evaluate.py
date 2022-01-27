from utils import *
from models import *
import os
import pandas as pd 
import numpy as np
import torch 
import transformers
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AdamW

DATASET_SUFFIX = "_fixed"
for dataset_type in ["test"]:
    if not os.path.isfile(DATASET_DIR + EMB_MODEL_CHECKPOINT + "_" + dataset_type + DATASET_SUFFIX+ ".csv"):
        print("[INFO] " + dataset_type + " dataset not found. Creating...")
        df = pd.read_csv(DATASET_DIR + dataset_type + ".csv")
        #df = df[:100] for testing 
        dataset_generator(df,EMB_MODEL_CHECKPOINT + "_" + dataset_type+DATASET_SUFFIX + ".csv")
        print("[INFO] " + dataset_type + " dataset created.")
    else:
        print("[INFO] " + dataset_type + " dataset found.")


test_df = pd.read_csv(DATASET_DIR + EMB_MODEL_CHECKPOINT + "_test" + DATASET_SUFFIX + ".csv")
test_dataset = CustomTextDataset(test_df)

model = BERTClass()
model.to(device)


test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
optim = AdamW(model.parameters(), lr=LEARNING_RATE)

output_model = SAVED_MODELS_DIR + "BERT_classifier"+ DATASET_SUFFIX+".bin"
load(model,optim,output_model)
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
    
get_benchmarks(y_true,y_pred)

