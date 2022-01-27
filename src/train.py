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
for dataset_type in ["train"]:
    if not os.path.isfile(DATASET_DIR + EMB_MODEL_CHECKPOINT + "_" + dataset_type + DATASET_SUFFIX+ ".csv"):
        print("[INFO] " + dataset_type + " dataset not found. Creating...")
        df = pd.read_csv(DATASET_DIR + dataset_type + ".csv")
        #df = df[:100] for testing 
        dataset_generator(df,EMB_MODEL_CHECKPOINT + "_" + dataset_type+DATASET_SUFFIX + ".csv")
        print("[INFO] " + dataset_type + " dataset created.")
    else:
        print("[INFO] " + dataset_type + " dataset found.")


train_df = pd.read_csv(DATASET_DIR + EMB_MODEL_CHECKPOINT + "_train" + DATASET_SUFFIX + ".csv")
train_dataset = CustomTextDataset(train_df)

model = BERTClass()
model.to(device)



train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
optim = AdamW(model.parameters(), lr=LEARNING_RATE)

running_loss = 0.0

for epoch in range(EPOCHS):
    for idx,batch in enumerate(tqdm(train_loader)):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)
        features = batch["features"].to(device)
        outputs = model(input_ids, attention_mask, token_type_ids,features)
        loss = loss_fn(outputs, labels)
        running_loss += loss 
        if idx % update_freq == 0:
            print("[INFO] Epoch: {} Loss : {}".format(epoch,running_loss/update_freq))
            running_loss = 0.0

        loss.backward()
        optim.step()
    output_model = SAVED_MODELS_DIR + "BERT_classifier"+ DATASET_SUFFIX+ "_" + str(epoch) +  ".bin"
    save(model, optim,output_model)
    print("[INFO] Model saved for epoch: {}".format(epoch))



output_model = SAVED_MODELS_DIR + "BERT_classifier"+ DATASET_SUFFIX+".bin"
save(model, optim,output_model)
print("[INFO] Model saved!")

