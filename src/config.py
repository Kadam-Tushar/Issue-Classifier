import argparse
from utils import get_device, get_args_dict

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)

    # Add user specific args
    parser.add_argument('--user', type=str, default = 'tusharpk')
    parser.add_argument('--project', type=str, default = 'test_project')

    # Add data args
    parser.add_argument('--dataset_type', type=str, default = 'test')
    parser.add_argument('--DATASET_SUFFIX', type=str, default = '_fixed')

    # Add model args
    parser.add_argument('--device', type=str, default = 'auto', help = 'gpu, cpu or auto')
    parser.add_argument('--EMB_MODEL_CHECKPOINT', type=str, default = 'bert-base-uncased')
    parser.add_argument('--MODEL_NAME', type=str, default = 'BERT')
    parser.add_argument('--TITLE_MAX_LEN', type=int, default = 100)
    parser.add_argument('--BATCH_SIZE', type=int, default = 32)
    parser.add_argument('--LEARNING_RATE', type=int, default = 5e-5)
    parser.add_argument('--EPOCHS', type=int, default = 3)
    parser.add_argument('--update_freq', type=int, default = 5000)
    parser.add_argument('--EARLY_ISSUE_THRESHOLD', type=int, default = 50)
    parser.add_argument('--dropout', type=float, default = 0.3)

    args, unparsed = parser.parse_known_args()
    if len(unparsed)>0:
        print(f'Warning: Unparsed arguments {unparsed}')

    args.device = get_device(args) #Get correct cuda device
    args.DATASET_DIR = get_home_path(args.user)
    args.SAVED_MODELS_DIR = args.DATASET_DIR + 'save/'
    args.LABEL_MAP = { "bug": 0, "enhancement": 1, "question": 2}
    args.INV_LABEL_MAP = {0: "bug", 1: "enhancement", 2: "question"}

    logging_args = get_args_dict(args, ignore_list = ['LABEL_MAP','INV_LABEL_MAP','device']) # For logging hyper-params on wandb server
    return args, logging_args


def get_home_path(user):
    if user == 'tusharpk':
        return '/data1/scratch/tusharpk/issue/'
    elif user == 'shikhar':
        return '/scratch/shikhar/Issue-Classifier/data/'