import torch
import torch.nn as nn
import transformers

class BERTClass(torch.nn.Module):
    def __init__(self, args):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(args.EMB_MODEL_CHECKPOINT).to(args.device)
        self.l2 = torch.nn.Dropout(args.dropout)
        # 768 is the output size of the bert model and extra 3 features are added from EDA
        self.l3 = torch.nn.Linear(768 + 3, 3)

    def forward(self, ids, mask, token_type_ids,features):
        out = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        out = self.l2(out[1])
        out = torch.cat((out,features),dim = -1)
        output = self.l3(out)
        return output