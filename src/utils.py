import pandas as pd 
import numpy as np
import torch 
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import *
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import sklearn.metrics
import ast

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

tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_CHECKPOINT)
model = AutoModel.from_pretrained(EMB_MODEL_CHECKPOINT).to(device)

# To convert any string into vectors
def text2vec(sentence):
    if "bert" in EMB_MODEL_CHECKPOINT:
        encoded_input = tokenizer(sentence, return_tensors='pt',padding=True, truncation=True).to(device)
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
def dataset_generator(orig_df,output_filename):
    
    # Code from EDA 
    modified_df = {'repository':[], 'repo_author':[], 'issue_number':[]}
    rep, auth, iss = process_urls(orig_df.repository_url, orig_df.issue_url)

    modified_df['repository'] = rep
    modified_df['repo_author'] = auth
    modified_df['issue_number'] = iss

    
    modified_df = pd.DataFrame(modified_df)
    modified_df['label'] = orig_df.issue_label.apply(lambda x: LABEL_MAP[x])
    modified_df['issue_author_association'] = orig_df.issue_author_association.tolist()
    modified_df['issue_title'] = orig_df.issue_title.tolist()
    modified_df['issue_body'] = orig_df.issue_body.tolist()
    modified_df["is_early_issue"] = modified_df.issue_number.apply(lambda x: 1 if x < EARLY_ISSUE_THRESHOLD else 0)
    modified_df["issue_body_length"] = modified_df.issue_body.apply(lambda x: len(x.split()) if type(x)!=float else 0)
    modified_df["is_opened_owner"] = modified_df.issue_author_association.apply(lambda x: 1 if x == "OWNER" else 0)
    
    # Preprocessing steps: tokenization of titles and 3 features from EDA : is_early_issue, issue_body_length, is_opened_owner
    
    modified_df["encodings"] = modified_df.issue_title.apply(lambda x: str(tokenizer(x,padding='max_length', truncation=True, max_length=TITLE_MAX_LEN)))
    modified_df["features"] = modified_df.apply(lambda x: str([x.is_early_issue, x.issue_body_length, x.is_opened_owner]), axis=1)
    modified_df = modified_df[["encodings","features", "label"]]
    modified_df.to_csv(DATASET_DIR + output_filename,index=False)


def get_benchmarks(y_true,y_pred):
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


def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)

# save model checkpoint
def save(model, optimizer,output_model):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_model)

# load model checkpoint
def load(model,optimizer,output_model):
    d = torch.load(output_model)
    model.load_state_dict(d["model_state_dict"])
    optimizer.load_state_dict(d["optimizer_state_dict"])

