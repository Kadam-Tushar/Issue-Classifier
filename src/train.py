import wandb
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
import os
from config import get_arguments
from utils import set_random_seed, CustomTextDataset, loss_fn, create_modified_dataset, save, load, get_free_gpus
from models import BERTClass
from evaluate import evaluate_model


def train(args):
    train_df = pd.read_csv(args.DATASET_DIR + args.EMB_MODEL_CHECKPOINT_NAME + "_train" + args.DATASET_SUFFIX + ".csv")
    train_dataset = CustomTextDataset(train_df)

    model = BERTClass(args)
    model.to(args.device)

    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
    optim = AdamW(model.parameters(), lr=args.LEARNING_RATE)

    running_loss = 0.0

    # Setting Summary metrics
    wandb.run.summary["best_P"] = 0.0
    wandb.run.summary["best_R"] = 0.0
    wandb.run.summary["best_F1"] = 0.0

    # Best epoch is epoch where we get the best F1 score
    wandb.run.summary["best_epoch"] = 0
    
    for epoch in range(args.EPOCHS):
        output_model = args.SAVED_MODELS_DIR + args.MODEL_NAME +"_classifier"+ args.DATASET_SUFFIX+ "_" + str(epoch) +  ".bin"
        should_train = True 

        # Check if trained model of current epoch exists already and if yes, load it. Else, train the model
        if os.path.isfile(output_model):
            print(f"[INFO] model {output_model} exists, skip training this epoch. Only Evaluate the model")
            load(model,optim,output_model)
            should_train = False
        

        if should_train:
            for idx,batch in enumerate(tqdm(train_loader)):
                optim.zero_grad()
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                #token_type_ids = batch['token_type_ids'].to(args.device) No need to pass token_type_ids as it is not used in BERTClass. 
                labels = batch['label'].to(args.device)
                features = batch["features"].to(args.device)
                outputs = model(input_ids, attention_mask,features)
                loss = loss_fn(outputs, labels)
                running_loss += loss.item() #Bug?
                if idx % args.update_freq == 0 and idx != 0:
                    print("[INFO] Epoch: {} Loss : {}".format(epoch,running_loss/args.update_freq))
                    wandb.log({"Loss": running_loss/args.update_freq})
                    running_loss = 0.0

                loss.backward()
                optim.step()

            save(model, optim, output_model)
            print("[INFO] Model saved for epoch: {}".format(epoch))
        
        # Evaluate model at every epoch
        P,R,F1 = evaluate_model(model, args)
        wandb.log({'Precision': P, 'Recall': R, 'F1': F1})
        wandb.run.summary["best_P"] = wandb.run.summary["best_P"] if wandb.run.summary["best_P"] > P else P
        wandb.run.summary["best_R"] = wandb.run.summary["best_R"] if wandb.run.summary["best_R"] > R else R
        wandb.run.summary["best_F1"] = wandb.run.summary["best_F1"] if wandb.run.summary["best_F1"] > F1 else F1
        wandb.run.summary["best_epoch"] = wandb.run.summary["best_epoch"] if wandb.run.summary["best_F1"] > F1 else epoch

        ##############################
        print("[INFO] Model evaluated and scores logged to server")
    
    wandb.run.summary["best_P"] = best_p
    wandb.run.summary["best_R"] = best_r
    wandb.run.summary["best_F1"] = best_f1

    output_model = args.SAVED_MODELS_DIR + args.MODEL_NAME + "_classifier"+ args.DATASET_SUFFIX+".bin"
    save(model, optim, output_model)
    print("[INFO] Model saved!")


def setup_visible_gpus():
    free_gpu_list = get_free_gpus(memory_req=20000,gpu_req=8)
    run_on_gpu = max(free_gpu_list)
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(run_on_gpu)
    print(f'[INFO] Running on cuda device {run_on_gpu}')


if __name__ == "__main__":
    setup_visible_gpus()
    args, logging_args = get_arguments()
    set_random_seed(args.seed, is_cuda = args.device == torch.device('cuda'))
    create_modified_dataset(args)
    wandb.init(project=args.project, entity="iisc1", config = logging_args)
    train(args)