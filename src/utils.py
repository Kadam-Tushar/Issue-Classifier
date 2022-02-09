import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from torch.utils.data import Dataset
import sklearn.metrics
import ast
import random

# class for Text Dataset that will be used in DataLoader API
class CustomTextDataset(Dataset):
    # Input is dataframe
    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        # item is dictionary of all the encodings, extra features for the text and label
        item = {key: torch.tensor(val) for key, val in ast.literal_eval(self.df.iloc[idx]["encodings"]).items()}
        item['features'] = torch.tensor(ast.literal_eval(self.df.iloc[idx]["features"]))
        item['label'] = torch.tensor(self.df.iloc[idx]["label"])
        return item

    def __len__(self):
        return len(self.df)


# To convert any string into vectors
def text2vec(sentence, args):
    if "bert" in args.EMB_MODEL_CHECKPOINT_NAME:
        tokenizer = AutoTokenizer.from_pretrained(args.EMB_MODEL_CHECKPOINT)
        model = AutoModel.from_pretrained(args.EMB_MODEL_CHECKPOINT).to(args.device)
        encoded_input = tokenizer(sentence, return_tensors='pt',padding=True, truncation=True , max_length = 512).to(args.device)
        output = model(**encoded_input)
        # First token is CLS token and its representation will contain information about sentence
        # So we will use it for classification tasks
        l = output.last_hidden_state[0,0,:].tolist()
        #l = l.reshape(768).tolist()
        return l


# Extract Repo Author, Repo Name and Issue number from URL
def process_urls(repo_url, issue_url):
    from urllib.parse import urlparse
    author = []
    repo = []
    issue_number = []
    for iss_url,rep_url in zip(issue_url,repo_url):
        assert iss_url.startswith(rep_url)
        parsed_url = urlparse(iss_url)
        url_path = parsed_url.path.split('/')
        auth, repo_name, _ ,iss_n = url_path[-4:]
        author.append(auth)
        repo.append(repo_name)
        issue_number.append(int(iss_n))
    return repo, author, issue_number


"""
Generates dataset after preprocessing, feature-engineering steps etc
orig_df : Dataframe  should contain all features given.

"""
def dataset_generator(orig_df, output_filename, args):

    tokenizer = AutoTokenizer.from_pretrained(args.EMB_MODEL_CHECKPOINT)

    # Code from EDA
    modified_df = {'repository':[], 'repo_author':[], 'issue_number':[]}
    rep, auth, iss = process_urls(orig_df.repository_url, orig_df.issue_url)

    modified_df['repository'] = rep
    modified_df['repo_author'] = auth
    modified_df['issue_number'] = iss


    modified_df = pd.DataFrame(modified_df)
    modified_df['label'] = orig_df.issue_label.apply(lambda x: args.LABEL_MAP[x])
    modified_df['issue_author_association'] = orig_df.issue_author_association.tolist()
    modified_df['issue_title'] = orig_df.issue_title.tolist()
    modified_df['issue_body'] = orig_df.issue_body.tolist()
    modified_df["issue_body"].replace(np.nan, '', inplace=True)
    modified_df["is_early_issue"] = modified_df.issue_number.apply(lambda x: 1 if x < args.EARLY_ISSUE_THRESHOLD else 0)
    modified_df["issue_body_length"] = modified_df.issue_body.apply(lambda x: len(x.split()) if type(x)!=float else 0)
    modified_df["is_opened_owner"] = modified_df.issue_author_association.apply(lambda x: 1 if x == "OWNER" else 0)
    modified_df["issue_text"] =  modified_df["issue_title"] + " " + modified_df["issue_body"]
    
    print("head of modified_df for debugging: ")
    print(modified_df["issue_text"].head())
    
    # Preprocessing steps: tokenization of titles and 3 features from EDA : is_early_issue, issue_body_length, is_opened_owner
    modified_df["encodings"] = modified_df.issue_text.apply(lambda x: str(tokenizer(x,padding='max_length', truncation=True, max_length=args.ISSUE_TEXT_MAX_LEN)))
    modified_df["features"] = modified_df.apply(lambda x: str([x.is_early_issue, x.issue_body_length, x.is_opened_owner]), axis=1)
    modified_df = modified_df[["encodings","features", "label"]]
    modified_df.to_csv(args.DATASET_DIR + output_filename,index=False)


def create_modified_dataset(args, dtype=['train']):
    for dataset_type in dtype:
        if not os.path.isfile(args.DATASET_DIR + args.EMB_MODEL_CHECKPOINT_NAME + "_" + dataset_type + args.DATASET_SUFFIX+ ".csv"):
            print("[INFO] " + dataset_type + " dataset not found. Creating...")
            df = pd.read_csv(args.DATASET_DIR + dataset_type + ".csv")
            # for testing pipeline on small dataset.
            #df = df[:100]  
            dataset_generator(df,args.EMB_MODEL_CHECKPOINT_NAME + "_" + dataset_type + args.DATASET_SUFFIX + ".csv", args)
            print("[INFO] " + dataset_type + " dataset created.")
        else:
            print("[INFO] " + dataset_type + " dataset found.")


def get_benchmarks(y_true,y_pred, INV_LABEL_MAP):
    for label in [0,1,2]:
        P_c = sklearn.metrics.precision_score(y_true, y_pred, average=None, labels=[label])[0]
        R_c = sklearn.metrics.recall_score(y_true, y_pred, average=None, labels=[label])[0]
        F1_c = sklearn.metrics.f1_score(y_true, y_pred, average=None, labels=[label])[0]
        print(f"=*= {INV_LABEL_MAP[label]} =*=")
        print(f"precision:\t{P_c:.4f}")
        print(f"recall:\t\t{R_c:.4f}")
        print(f"F1 score:\t{F1_c:.4f}")
        print()

    P = sklearn.metrics.precision_score(y_true, y_pred, average='micro')
    R = sklearn.metrics.recall_score(y_true, y_pred, average='micro')
    F1 = sklearn.metrics.f1_score(y_true, y_pred, average='micro')

    print("=*= global =*=")
    print(f"precision:\t{P:.4f}")
    print(f"recall:\t\t{R:.4f}")
    print(f"F1 score:\t{F1:.4f}")
    return (P,R,F1)


def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)


def check_path(path, overwrite=False):
    """Check if `path` exists, makedirs if not else warning/IOError."""
    directory = os.path.dirname(path)
    if os.path.exists(path):
        if not overwrite:
            raise IOError(f"[ERROR] path {path} exists, stop.")
    else:
        if not os.path.exists(directory):
            print(f'[INFO] created directory {directory}')
        os.makedirs(directory, exist_ok=True)


# save model checkpoint
def save(model, optimizer, output_model_path):
    # save
    check_path(output_model_path, overwrite=True) 
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_model_path)


# load model checkpoint
def load(model,optimizer,output_model):
    d = torch.load(output_model)
    model.load_state_dict(d["model_state_dict"])
    optimizer.load_state_dict(d["optimizer_state_dict"])


##############################################################
def get_device(args):
    import torch
    if args.device not in ['gpu','cpu','auto']:
        print(f'[ERROR] Device option not supported. Received {args.device}')
        return args.device
    if(args.device == 'gpu' and not torch.cuda.is_available()):
        print('[WARNING] Backend device: %s not available. Will default to auto choice.',args.device)
        args.device = 'auto'

    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # auto choice - always use cuda if available
    elif args.device == 'cpu': # unless explicitly requested
        device = torch.device('cpu')
    elif args.device == 'gpu':
        device = torch.device("cuda")
    args.device=device
    print(f'[INFO] Using device {args.device}')
    return device


def set_random_seed(seed, is_cuda):
    """Sets the random seed."""
    if seed > 0:
        torch.manual_seed(seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)

    if is_cuda and seed > 0:
        # These ensure same initialization in multi gpu mode
        torch.cuda.manual_seed(seed)


def get_args_dict(obj, ignore_list = []):
    import inspect
    prop = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not inspect.ismethod(value) and not name in ignore_list:
            prop[name] = value
    return prop


def get_free_gpus(memory_req=15000,gpu_req=1):
    import subprocess as sp
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [(i,int(x.split()[0])) for i, x in enumerate(memory_free_info)]
    free_gpu_list = []
    for gpu_id,gpu_memory in memory_free_values:
        if gpu_memory>=memory_req:
            free_gpu_list.append(gpu_id)
            gpu_req-=1
            if gpu_req==0:
                break
    return free_gpu_list