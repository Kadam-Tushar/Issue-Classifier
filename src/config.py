import argparse
from utils import get_device, get_args_dict
import torch
import json 
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)

    # Add user specific args
    parser.add_argument('--user', type=str, default = 'tusharpk')
    parser.add_argument('--project', type=str, default = 'test_project')

    # Add data args
    parser.add_argument('--dataset_type', type=str, default = 'test')
    parser.add_argument('--DATASET_SUFFIX', type=str, default = '_body_added')

    # Add model args
    parser.add_argument('--device', type=str, default = 'auto', help = 'gpu, cpu or auto')
    parser.add_argument('--EMB_MODEL_CHECKPOINT', type=str, default = 'bert-base-uncased')
    parser.add_argument('--MODEL_NAME', type=str, default = 'BERT')
    parser.add_argument('--TITLE_MAX_LEN', type=int, default = 100)
    parser.add_argument('--ISSUE_TEXT_MAX_LEN', type=int, default = 512)
    parser.add_argument('--BATCH_SIZE', type=int, default = 16)
    parser.add_argument('--LEARNING_RATE', type=float, default = 2.1834022685908154e-05)
    parser.add_argument('--EPOCHS', type=int, default = 4)
    parser.add_argument('--update_freq', type=int, default = 5000)
    parser.add_argument('--EARLY_ISSUE_THRESHOLD', type=int, default = 98)
    parser.add_argument('--dropout', type=float, default = 0.2421181906958028)

    args, unparsed = parser.parse_known_args()

    """
    Use EMB_MODEL_CHECKPOINT only to save and load model from hugging face from_pretrained() save_pretrained() methods.
    Use EMB_MODEL_CHECKPOINT_NAME as part of unique name of model for saving model/dataset on disc.
    HF model names have '/' in them, so we replace '/' with '-' in EMB_MODEL_CHECKPOINT_NAME
    """
    args.EMB_MODEL_CHECKPOINT_NAME = args.EMB_MODEL_CHECKPOINT.replace("/","-")
    if len(unparsed)>0:
        print(f'Warning: Unparsed arguments {unparsed}')
    
    # Setting index of cuda device 
    torch.cuda.set_device(6)
    args.device = get_device(args) #Get correct cuda device
    args.DATASET_DIR = get_data_dir_path(args.user)
    args.SAVED_MODELS_DIR = args.DATASET_DIR + 'save/'
    args.LABEL_MAP = { "bug": 0, "enhancement": 1, "question": 2}
    args.INV_LABEL_MAP = {0: "bug", 1: "enhancement", 2: "question"}

    logging_args = get_args_dict(args, ignore_list = ['LABEL_MAP','INV_LABEL_MAP','device']) # For logging hyper-params on wandb server
    print("Logging parsed params:")
    print(json.dumps(logging_args,sort_keys=False, indent=4))
    return args, logging_args


def get_data_dir_path(user):
    if user == 'tusharpk':
        return '/data1/scratch/tusharpk/issue/'
    elif user == 'shikhar':
        return '/data1/scratch/shikhar/IssueClassifierRuns/'
