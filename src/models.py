import torch
import torch.nn as nn
import transformers

class BERTClass(torch.nn.Module):
    def __init__(self, args):
        super(BERTClass, self).__init__()
        self.l1 = transformers.AutoModel.from_pretrained(args.EMB_MODEL_CHECKPOINT).to(args.device)
        self.l2 = torch.nn.Dropout(args.dropout)
        # Output size is 1024 for bert-large and 768 for bert-base models and 3*embedding_size for features
        output_size = 1024 if "large" in args.EMB_MODEL_CHECKPOINT else 768 
        embedding_size = 8
        self.feature_embedding_layer = nn.ModuleList([nn.Embedding(2, embedding_size) for _ in range(3)])
        self.l3 = torch.nn.Linear(output_size + 3*embedding_size, 3)

    def forward(self, ids, mask, features):
        out = self.l1(ids, attention_mask = mask)
        """
        Extracting only [CLS] token embedding of last_hidden_state
        out[0]: last_hidden_state with dim [batch_size, seq_len, hidden_size]
        dimensions( out[0][:,0,:]) =  [batch_size, hidden_size] of [CLS] token
        """
        out = self.l2(out[0][:,0,:])
        features = torch.cat([self.feature_embedding_layer[i](features[:,i]) for i in range(3)], dim=-1)
        out = torch.cat((out,features),dim = -1) # Concatenate embedded features
        output = self.l3(out)
        return output
    
    # Comment out this to make HF model in eval mode always
    # def train(self,mode=True):
    #     super().train(mode)
    #     self.l1.eval()

    def get_emb(self, ids, mask, features):
        out = self.l1(ids, attention_mask = mask)
        out = self.l2(out[0][:,0,:])
        features = torch.cat([self.feature_embedding_layer[i](features[:,i]) for i in range(3) if i == 0], dim=-1)
        out = torch.cat((out,features),dim = -1)
        return out
