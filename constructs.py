"""
Reserve index 0 in the vocabulary for padding.

Write additional code that saves a vocabulary, and converts the dataset into integer index and integer index target. DO NOT PAD IT
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence

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

        tmp_output, _, _=self.lstm(x)
        tmp_output=pad_packed_sequence(tmp_output, batch_first=self.batch_first)
        a, b=tmp_output[:, :, :self.embed_dim], tmp_output[:, :, self.embed_dim]

        if self.use_softmax:
            return self.softmax(a+b)
        
        return a+b

class ner_dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_loc, vocab_size, max_seq_len=40):
        
        super(ner_dataset, self).__init__()
        
        if vocab_size>65536:
            tmp_data=pd.read_csv(dataset_loc, header=None, dtype=np.uint32).values
        else:
            tmp_data=pd.read_csv(dataset_loc, header=None, dtype=np.uint16).values

        unpad_x, target=tmp_data[:, :-1], tmp_data[:, -1]
        batch_sizes=unpad_x.shape[1]-np.sum(np.isnan(unpad_x), axis=1)

        # now, just convert the nan to 0 to pad
        # Note again, vocal must have index 0 reserved for pad

        unpad_x[np.isnan(unpad_x)]=0

        self.unpad_x=torch.from_numpy(unpad_x)
        self.ner_targets=torch.from_numpy(target)
        self.batch_sizes=torch.from_numpy(batch_sizes)

        return

    def __len__(self):

        return len(self.unpad_x)

    def __getitem__(self, idx):

        return self.unpad_x[idx], self.batch_first[idx], self.ner_targets[idx]

args={}

args{"logging_dir"}=""
args{"epochs"}=30
args["optimizer"]=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3)
args["lr_schedulers"]=[lambda opt: torch.optim.lr_scheduler.StepLR(opt, 9, gamma=0.8)]
args{"batch_first"}=True
args{"device"}="cuda"
args{"embedding_dim"}=50
