import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

class NER_Model(nn.Module):

    def __init__(self, lang_model, vocab_size, embedding_dim=50):

        """
        lang_model: The neural network that processes the token vector sequence.
        vocab_size: Int, the size of the vocaburary.
        embedding_dim: The dimension of the embedding vector space. Default, 50.

        Note about the lang_model: Given a batch of (embedding transformed) & packed sequence sentences (the shape of the input is the whatever is supplied by the dataloader),
        the lang_model must return a tuple that has output (N, max_seq_len, num_ner_classes) if batch is first,
        else (max_seq_len, N, num_ner_classes) if batch is first is False
        """

        super(NER_Model, self).__init__()
        self.lang_model=lang_model
        self.embed_dim=embedding_dim
        self.embedding=nn.Embedding(vocab_size, self.embed_dim)

        return

    def forward(self, x, batch_sizes, batch_first=True):

        """
        x is expected to be a padded tensor
        x should NOT be a packed sequence, that will be done here using batch_sizes 
        x is expected to have exactly 2 dimensions. One will be batch, the other the words of the sentence.
    
        Suggested implementation of the trainloader: Output padded x, the sequence_lenghts and the target as padded sequence of integer index of NER class

        If batch_first is True, x has shape (N, max_seq_len) and (max_seq_len, N) otherwise
        A sentence is expected to be a sequence of ints, representing the index in the vocabs
        """

        num_rows, num_cols=x.shape
        x_embed=pack_padded_sequence(self.embedding(x.to(torch.int32).flatten()).reshape((num_rows, num_cols, -1)), batch_sizes, batch_first=batch_first, enforce_sorted=False)

        return self.lang_model(x_embed)