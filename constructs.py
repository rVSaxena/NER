"""
Reserve index 0 in the vocabulary for padding.

Write additional code that saves a vocabulary, and converts the dataset into integer index and integer index target. DO NOT PAD IT
"""

import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class LangModel(nn.Module):

    """
    Given a batch of (embedding transformed) & packed sequences (sentences) (the shape of the input is the whatever is supplied by the dataloader),
    the lang_model must return a tuple that has output (N, max_seq_len, num_ner_classes) if batch is first,
    else (max_seq_len, N, num_ner_classes) if batch is first is False
    """

    def __init__(self, embed_dim, num_classes, batch_first=True, use_softmax=False):

        super(LangModel, self).__init__()
        self.batch_first=batch_first
        self.embed_dim=embed_dim
        self.num_classes=num_classes
        self.lstm=nn.LSTM(
            input_size=embed_dim,
            hidden_size=128,
            proj_size=num_classes,
            bidirectional=True,
            batch_first=self.batch_first,
            dropout=0.3,
            num_layers=3
            )
        self.use_softmax=use_softmax
        if self.use_softmax:
            self.softmax=nn.Softmax(dim=2)

        return

    def forward(self, x):
        """
        Should expect a packed, padded sequence of embedding transformed sentences.
        Shape of underlying data is (N, max_seq_len, embed_dim) if batch_first
        else (max_seq_len, N, embed_dim)
        """

        tmp_output, _=self.lstm(x)
        tmp_output, tmp_batch_sizes=pad_packed_sequence(tmp_output, batch_first=self.batch_first)
        a, b=tmp_output[:, :, :self.num_classes], tmp_output[:, :, self.num_classes:]

        if self.use_softmax:
            return pack_padded_sequence(self.softmax(a+b), tmp_batch_sizes, batch_first=self.batch_first, enforce_sorted=False)
        
        return pack_padded_sequence(a+b, tmp_batch_sizes, batch_first=self.batch_first, enforce_sorted=False)

class ner_dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_loc, target_loc, vocab_size, max_seq_len=40):
        
        super(ner_dataset, self).__init__()
        
        unpad_x=pd.read_csv(dataset_loc).values
        target=pd.read_csv(target_loc).values

        batch_sizes=unpad_x.shape[1]-np.sum(np.isnan(unpad_x), axis=1)

        # now, just convert the nan to 0 to pad
        # Note again, vocal must have index 0 reserved for pad

        unpad_x[np.isnan(unpad_x)]=0
        target[np.isnan(target)]=100

        self.pad_x=torch.from_numpy(unpad_x).to(torch.int32)
        self.ner_targets=torch.from_numpy(target).to(torch.int32)
        self.batch_sizes=torch.from_numpy(batch_sizes).to(torch.int32)

        return

    def __len__(self):

        return len(self.pad_x)

    def __getitem__(self, idx):

        return self.pad_x[idx], self.batch_sizes[idx], self.ner_targets[idx]

args={}

args["logging_dir"]="./NER_run2"
args["target_pad_value"]=100
args["epochs"]=30
args["optimizer"]=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3)
args["lr_schedulers"]=[lambda opt: torch.optim.lr_scheduler.StepLR(opt, 9, gamma=0.8)]
args["batch_first"]=True
args["device"]="cuda"
args["embedding_dim"]=50
args["max_seq_len"]=104
args["num_classes"]=17
args["vocab_size"]=36039

dataset=ner_dataset("run2Data/data.csv", "run2Data/labels.csv", args["vocab_size"], args["max_seq_len"])
trainloader=torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

lang_model=LangModel(args["embedding_dim"], args["num_classes"])

args["trainLoader"]=trainloader
args["lang_model"]=lang_model