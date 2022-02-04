import wandb
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW

from config import get_arguments
from utils import set_random_seed, CustomTextDataset, loss_fn, create_modified_dataset, save
from models import BERTClass
from evaluate import evaluate_model


def train(args):
    train_df = pd.read_csv(args.DATASET_DIR + args.EMB_MODEL_CHECKPOINT + "_train" + args.DATASET_SUFFIX + ".csv")
    train_dataset = CustomTextDataset(train_df)

    model = BERTClass(args)
    model.to(args.device)

    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
    optim = AdamW(model.parameters(), lr=args.LEARNING_RATE)

    running_loss = 0.0

    for epoch in range(args.EPOCHS):
        for idx,batch in enumerate(tqdm(train_loader)):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            token_type_ids = batch['token_type_ids'].to(args.device)
            labels = batch['label'].to(args.device)
            features = batch["features"].to(args.device)
            outputs = model(input_ids, attention_mask, token_type_ids,features)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() #Bug?
            if idx % args.update_freq == 0:
                print("[INFO] Epoch: {} Loss : {}".format(epoch,running_loss/args.update_freq))
                wandb.log({"Loss": running_loss/args.update_freq})
                running_loss = 0.0
                break

            loss.backward()
            optim.step()
        output_model = args.SAVED_MODELS_DIR + "BERT_classifier"+ args.DATASET_SUFFIX+ "_" + str(epoch) +  ".bin"
        save(model, optim, output_model)
        print("[INFO] Model saved for epoch: {}".format(epoch))
        P,R,F1 = evaluate_model(model, args)
        wandb.log({'Precision': P, 'Recall': R, 'F1': F1})
        print("[INFO] Model evaluated and scores logged to server")

    output_model = args.SAVED_MODELS_DIR + "BERT_classifier"+ args.DATASET_SUFFIX+".bin"
    save(model, optim, output_model)
    print("[INFO] Model saved!")


if __name__ == "__main__":
    args, logging_args = get_arguments()
    set_random_seed(args.seed, is_cuda = args.device == torch.device('cuda'))
    create_modified_dataset(args)
    wandb.init(project=args.project, entity="iisc1")
    wandb.config = logging_args
    print('Logging following arguments', logging_args)
    train(args)