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
from CRF import CRF
from os import makedirs


batch_first=args["batch_first"]
device=args["device"]
trainLoader=args["trainLoader"] # a torch.utils.data.DataLoader object
epochs=args["epochs"]

if "embedding_dim" in args:
    ner_features_model=NER_Model(
        args["lang_model"],
        args["vocab_size"]
        ).to(device)
else:
    ner_features_model=NER_Model(
        args["lang_model"],
        args["vocab_size"],
        args["embedding_dim"]
        ).to(device)

crf=CRF(
    args["num_classes"],
    args["batch_first"],
    args["target_pad_value"],
    device
    ).to(device)

model=nn.ModuleList([ner_features_model, crf])
model.train()
optimizer=args["optimizer"](list(model.parameters()))
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
                ner_pred=model[0](batch_x, batch_seq_lens, batch_first) # this is a packed sequence
                packed_labels=pack_padded_sequence(
                    batch_labels,
                    batch_seq_lens,
                    args["batch_first"],
                    enforce_sorted=False
                    )
                batch_loss=-model[1].score_tag_sequence(ner_pred, packed_labels)
                batch_loss.backward()
                lossarr.append(batch_loss.item())
                
                optimizer.step()

                t.set_postfix(crf_score=batch_loss.item())

            for schs in lr_schedulers:
                schs.step()

        torch.save(model.state_dict(), pathjoin(args["logging_dir"], "models", "{}.pth".format(epoch)))
        np.savetxt(pathjoin(args["logging_dir"], "loss_values", "epoch_{}.csv".format(epoch)), lossarr, delimiter=",")

    torch.save(model.state_dict(), pathjoin(args["logging_dir"], "final_model.pth"))
    print("Done! Model saved at {}".format(pathjoin(args["logging_dir"], "model.pth")))