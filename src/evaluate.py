import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import BERTClass
from utils import get_benchmarks, create_modified_dataset, load, set_random_seed, CustomTextDataset
from config import get_arguments


def load_model(args):
    model = BERTClass()
    model.to(args.device)
    output_model = args.SAVED_MODELS_DIR + args.MODEL_NAME + "_classifier"+ args.DATASET_SUFFIX+".bin"
    load(model, output_model)
    return model


def evaluate_model(model, args):
    create_modified_dataset(args, dtype = ['test'])
    test_df = pd.read_csv(args.DATASET_DIR + args.EMB_MODEL_CHECKPOINT_NAME + "_test" + args.DATASET_SUFFIX + ".csv")
    test_dataset = CustomTextDataset(test_df)

    test_loader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
    

    model.eval()
    print("[INFO] Model loaded!")

    # Evaluating the model
    y_true = []
    y_pred = []
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        #token_type_ids = batch['token_type_ids'].to(args.device)
        features = batch["features"].to(args.device)
        labels = batch['label'].to(args.device)
        outputs = model(input_ids, attention_mask,features)
        y_true += labels.cpu().detach().numpy().tolist()
        y_pred += torch.argmax(outputs, dim=1).cpu().detach().numpy().tolist()

    return get_benchmarks(y_true,y_pred, args.INV_LABEL_MAP)


if __name__ == '__main__':
    args, logging_args = get_arguments()
    set_random_seed(args.seed, is_cuda = args.device == torch.device('cuda'))
    model = load_model(args)
    _ = evaluate_model(model, args)