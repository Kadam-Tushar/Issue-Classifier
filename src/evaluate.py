import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import BERTClass
from utils import get_benchmarks, create_modified_dataset, load, set_random_seed, CustomTextDataset
from config import get_arguments


def load_model(args):
    model = BERTClass(args)
    model.to(args.device)
    output_model_name = args.SAVED_MODELS_DIR + args.MODEL_NAME +"_classifier"+ args.DATASET_SUFFIX+ "_" + 'best' +  ".bin"
    load(model, output_model_name)
    return model


def evaluate_model(model, args, split='test', limit_examples=None):
    create_modified_dataset(args, dtype = [split])
    eval_df = pd.read_csv(args.DATASET_DIR + args.EMB_MODEL_CHECKPOINT_NAME + "_" + split + args.DATASET_SUFFIX + ".split.csv")
    eval_dataset = CustomTextDataset(eval_df)

    eval_loader = DataLoader(eval_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
    

    model.eval()
    print("[INFO] Model loaded!")

    # Evaluating the model
    y_true = []
    y_pred = []
    idx=0
    for batch in tqdm(eval_loader):
        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        #token_type_ids = batch['token_type_ids'].to(args.device)
        features = batch["features"].to(args.device)
        labels = batch['label'].to(args.device)
        outputs = model(input_ids, attention_mask,features)
        y_true += labels.cpu().detach().numpy().tolist()
        y_pred += torch.argmax(outputs, dim=1).cpu().detach().numpy().tolist()
        if limit_examples is not None:
            idx+=1
            if idx >= limit_examples:
                break

    return get_benchmarks(y_true,y_pred, args.INV_LABEL_MAP)


if __name__ == '__main__':
    args, logging_args = get_arguments()
    set_random_seed(args.seed, is_cuda = args.device == torch.device('cuda'))
    model = load_model(args)
    _ = evaluate_model(model, args, split = 'test')
