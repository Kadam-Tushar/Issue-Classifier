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
import re
import nltk.corpus
from nltk.corpus import nps_chat
import pandas as pd
import re

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


# To detect if setence is question or not using nlp : taken from https://github.com/kartikn27/nlp-question-detection
class IsQuestion():

    # Init constructor
    def __init__(self):
        posts = self.__get_posts()
        feature_set = self.__get_feature_set(posts)
        self.classifier = self.__perform_classification(feature_set)

    # Method (Private): __get_posts
    # Input: None
    # Output: Posts (Text) from NLTK's nps_chat package
    def __get_posts(self):
        return nltk.corpus.nps_chat.xml_posts()

    # Method (Private): __get_feature_set
    # Input: Posts from NLTK's nps_chat package
    # Processing: 1. preserve alpha numeric characters, whitespace, apostrophe
    # 2. Tokenize sentences using NLTK's word_tokenize
    # 3. Create a dictionary of list of tuples for each post where tuples index 0 is the dictionary of words occuring in the sentence and index 1 is the class as received from nps_chat package
    # Return: List of tuples for each post
    def __get_feature_set(self, posts):
        feature_list = []
        for post in posts:
            post_text = post.text
            features = {}
            words = nltk.word_tokenize(post_text)
            for word in words:
                features['contains({})'.format(word.lower())] = True
            feature_list.append((features, post.get('class')))
        return feature_list

    # Method (Private): __perform_classification
    # Input: List of tuples for each post
    # Processing: 1. Divide data into 80% training and 10% testing sets
    # 2. Use NLTK's Multinomial Naive Bayes to perform classifcation
    # 3. Print the Accuracy of the model
    # Return: Classifier object
    def __perform_classification(self, feature_set):
        training_size = int(len(feature_set) * 0.1)
        train_set, test_set = feature_set[training_size:], feature_set[:training_size]
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        print('Accuracy is : ', nltk.classify.accuracy(classifier, test_set))
        return classifier

    # Method (private): __get_question_words_set
    # Input: None
    # Return: Set of commonly occuring words in questions
    def __get_question_words_set(self):
        question_word_list = ['what', 'where', 'when','how','why','did','do','does','have','has','am','is','are','can','could','may','would','will','should'
"didn't","doesn't","haven't","isn't","aren't","can't","couldn't","wouldn't","won't","shouldn't",'?']
        return set(question_word_list)

    # Method (Public): predict_question
    # Input: Sentence to be predicted
    # Return: 1 - If sentence is question | 0 - If sentence is not question
    def predict_question(self, text):
        words = nltk.word_tokenize(text.lower())
        if self.__get_question_words_set().intersection(words) == False:
            return 0
        if '?' in text:
            return 1

        features = {}
        for word in words:
            features['contains({})'.format(word.lower())] = True

        prediction_result = self.classifier.classify(features)
        if prediction_result == 'whQuestion' or prediction_result == 'ynQuestion':
            return 1
        return 0

    # Method (Public): predict_question_type
    # Input: Sentence to be predicted
    # Return: 'WH' - If question is WH question | 'YN' - If sentence is Yes/NO question | 'unknown' - If unknown question type
    def predict_question_type(self, text):
        words = nltk.word_tokenize(text.lower())
        features = {}
        for word in words:
            features['contains({})'.format(word.lower())] = True

        prediction_result = self.classifier.classify(features)
        if prediction_result == 'whQuestion':
            return 'WH'
        elif prediction_result == 'ynQuestion':
            return 'YN'
        else:
            return 'unknown'
isQ = IsQuestion()

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


def clean_body(text):
    # Cleans github issue body text
    text = re.sub(r'```(.*?)```','[CODE]',text) # replace github code with [CODE]
    text = re.sub(r'https?://\S+|www\.\S+','[URL]',text) # replace urls with [URL]
    text = re.sub(r'@\S+','[USER]',text) # replace @user with [USER]
    text = re.sub(r'^>','',text) #Remove > symbol from start of line
    text = re.sub(r'\d+','[NUMBER]',text) # replace numbers with [NUMBER]
    text = re.sub(r'`(.*?)`','[CODE_INLINE]',text) # replace in line code blocks with [CODE_INLINE]
    text = re.sub(r'\n',' ',text) # replace new line with space
    text = re.sub(r'\s+',' ',text) # replace multiple spaces with single space
    return text


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
    modified_df["is_opened_owner"] = modified_df.issue_author_association.apply(lambda x: 1 if x == "OWNER" else 0)

    modified_df["is_question"] = modified_df.issue_title.apply(lambda x : isQ.predict_question(x))

    modified_df["issue_text"] =  modified_df["issue_title"] + " [B] " + modified_df["issue_body"].apply(clean_body)

    print("head of modified_df for debugging: ")
    print(modified_df["issue_text"].head())

    # Preprocessing steps: tokenization of titles and 3 features from EDA : is_early_issue, is_opened_owner
    modified_df["encodings"] = modified_df.issue_text.apply(lambda x: str(tokenizer(x,padding='max_length', truncation=True, max_length=args.ISSUE_TEXT_MAX_LEN)))
    modified_df["features"] = modified_df.apply(lambda x: str([x.is_early_issue, x.is_opened_owner, x.is_question]), axis=1)
    modified_df = modified_df[["encodings","features", "label"]]
    modified_df.to_csv(args.DATASET_DIR + output_filename,index=False)


def create_modified_dataset(args, dtype=['train']):
    for dataset_type in dtype:
        if not os.path.isfile(args.DATASET_DIR + args.EMB_MODEL_CHECKPOINT_NAME + "_" + dataset_type + args.DATASET_SUFFIX+ ".split.csv"):
            print("[INFO] " + dataset_type + " dataset not found. Creating...")
            df = pd.read_csv(args.DATASET_DIR + dataset_type + ".split.csv")
            dataset_generator(df,args.EMB_MODEL_CHECKPOINT_NAME + "_" + dataset_type + args.DATASET_SUFFIX + ".split.csv", args)
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


def loss_fn(outputs, targets, weight = None):
    if weight is not None:
        weight_vec = torch.Tensor(weight).type_as(outputs)
        return torch.nn.CrossEntropyLoss(weight = weight_vec)(outputs, targets)
    else:
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