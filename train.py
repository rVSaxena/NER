"""
Supports 1 optimizer and multiple LR schedulers on that optimizer
"""

import torch
import numpy as np
from tqdm import tqdm
from os.path import join as pathjoin
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import CrossEntropyLoss
from constructs import args
from ner_model import NER_Model
from os import makedirs

def loss_func(target_pad_value, max_seq_len, num_classes, batch_first=False):

    lf=CrossEntropyLoss(ignore_index=target_pad_value)

    def f(ner_pred, ner_target):
        """
        ner_pred: Is a packed_sequence. Underlying data should have shape (max_seq_len, N, num_ner_classes) if not batch_first, else
        (N, max_seq_len, num_ner_pred) if batch_first
        ner_target: Is a padded tensor of shape (max_seq_len, N) if not batch_first, else (N, max_seq_len) if batch_first
        """

        ner_pred=pad_packed_sequence(ner_pred, batch_first=batch_first, total_length=max_seq_len)[0].reshape((-1, num_classes))
        ner_target=ner_target.reshape((-1))
        return lf(ner_pred, ner_target)

    return f

batch_first=args["batch_first"]
loss=loss_func(args["target_pad_value"], args["max_seq_len"], args["num_classes"], batch_first)

device=args["device"]
trainLoader=args["trainLoader"] # a torch.utils.data.DataLoader object
epochs=args["epochs"]

if "embedding_dim" in args:
    model=NER_Model(
        args["lang_model"],
        args["vocab_size"]
        ).to(device)
else:
    model=NER_Model(
        args["lang_model"],
        args["vocab_size"],
        args["embedding_dim"]
        ).to(device)

model.train()
optimizer=args["optimizer"](model)
lr_schedulers=[f(optimizer) for f in args["lr_schedulers"]]

makedirs(pathjoin(args["logging_dir"], "models"), exist_ok=True)
makedirs(pathjoin(args["logging_dir"], "loss_values"), exist_ok=True)

if __name__=='__main__':
    
    for epoch in range(epochs):

        lossarr=[]
        
        with tqdm(trainLoader) as t:

            t.set_description("Epoch: {}".format(epoch))
        
            for batch_x, batch_seq_lens, batch_labels  in t:

                optimizer.zero_grad()

                batch_x, batch_seq_lens, batch_labels=batch_x.to(device), batch_seq_lens.to(torch.int64).to('cpu'), batch_labels.type(torch.LongTensor).to(device)
                ner_pred=model(batch_x, batch_seq_lens, batch_first)
                batch_loss=loss(ner_pred, batch_labels)
                batch_loss.backward()
                lossarr.append(batch_loss.item())
                
                optimizer.step()

                t.set_postfix(cross_entropy_loss=batch_loss.item())

            for schs in lr_schedulers:
                schs.step()

        torch.save(model.state_dict(), pathjoin(args["logging_dir"], "models", "{}.pth".format(epoch)))
        np.savetxt(pathjoin(args["logging_dir"], "loss_values", "epoch_{}.csv".format(epoch)), lossarr, delimiter=",")

    torch.save(model.state_dict(), pathjoin(args["logging_dir"], "final_model.pth"))
    print("Done! Model saved at {}".format(pathjoin(args["logging_dir"], "model.pth")))