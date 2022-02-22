import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluate import load_model
from models import BERTClass
from utils import get_benchmarks, create_modified_dataset, load, set_random_seed, CustomTextDataset
from config import get_arguments
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_emb(model, args, split='test', limit_examples=None):
    create_modified_dataset(args, dtype = [split])
    eval_df = pd.read_csv(args.DATASET_DIR + args.EMB_MODEL_CHECKPOINT_NAME + "_" + split + args.DATASET_SUFFIX + ".split.csv")
    eval_dataset = CustomTextDataset(eval_df)

    eval_loader = DataLoader(eval_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
    

    model.eval()
    print("[INFO] Model loaded!")

    # Getting learned embeddings. 
    emb = []
    y_true = []
    idx=0
    for batch in tqdm(eval_loader):
        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        #token_type_ids = batch['token_type_ids'].to(args.device)
        features = batch["features"].to(args.device)
        labels = batch['label'].to(args.device)
        outputs = model.get_emb(input_ids, attention_mask,features)
        y_true += labels.cpu().detach().numpy().tolist()
        emb += outputs.cpu().detach().numpy().tolist()
        if limit_examples is not None:
            idx+=1
            if idx >= limit_examples:
                break

    test_df = pd.DataFrame(emb)
    #print(y_true)
    
    tsne_embedded = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(test_df)
    pca_embedded = PCA(n_components=2).fit_transform(test_df)
    df = pd.DataFrame(tsne_embedded,columns=['x-cord','y-cord'])
    df['classes'] = y_true
    df['classes'] = df['classes'].apply(lambda x: args.INV_LABEL_MAP[x])
    
    sns.scatterplot(data=df, x="x-cord", y="y-cord", hue="classes" , palette = sns.color_palette("tab10",n_colors=3)).set_title('t-SNE plot')
    
    plt.savefig("tsne_plots.png")

    """ Comment out below code for pca plot. """
    # df = pd.DataFrame(pca_embedded,columns=['x-cord','y-cord'])
    # df['classes'] = y_true
    # df['classes'] = df['classes'].apply(lambda x: args.INV_LABEL_MAP[x])
    # sns.scatterplot(data=df, x="x-cord", y="y-cord", hue="classes" ,  palette = sns.color_palette("tab10",n_colors=3) ).set_title('PCA plots')
  

    # plt.savefig("pca_plots.png")
    

if __name__ == '__main__':
    args, logging_args = get_arguments()
    set_random_seed(args.seed, is_cuda = args.device == torch.device('cuda'))
    model = load_model(args)
    visualize_emb(model, args, split = 'test',limit_examples = 50)