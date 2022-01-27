import torch 

EMB_MODEL_CHECKPOINT = "bert-base-uncased"
DATASET_DIR = "/data1/scratch/tusharpk/issue/"  
LABEL_MAP = { "bug": 0, "enhancement": 1, "question": 2}
INV_LABEL_MAP = {0: "bug", 1: "enhancement", 2: "question"}
dataset_type = "test"

MODEL_NAME = "BERT"
SAVED_MODELS_DIR = DATASET_DIR + "save/" 

TITLE_MAX_LEN = 100
torch.cuda.set_device(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
EPOCHS = 3
update_freq = 5000
EARLY_ISSUE_THRESHOLD = 50 