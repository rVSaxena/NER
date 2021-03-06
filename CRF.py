import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class CRF(nn.Module):

    """
    Implements linear chain CRFs. Vectorized operations
    keep it fast. 

    Implementation relies on the X (or features) being padded with 0.
    Padding with other values will give un-tested and un-known results.

    If unfamiliar, http://www.cs.columbia.edu/~mcollins/crf.pdf is a good resource.
    """

    # This implementation does not use an <end> tag.
    # Can be changed to use, but no use yet.
    

    def __init__(self, num_classes, batch_first, target_pad_val, device):
        
        """
        Parameters:
            num_classes:
            batch_first:
            target__pad_val:
            device: The device that any newly created tensors will reside on.
        """

        super(CRF, self).__init__()

        self.transition_scores=nn.parameter.Parameter(data=torch.rand(num_classes, num_classes))
        self.origination_scores=nn.parameter.Parameter(data=torch.rand(num_classes, ))
        self.target_pad_val=target_pad_val
        self.batch_first=batch_first
        self.num_classes=num_classes
        self.device=device

        return

    def decode(self, x):

        """
        Because decode is used for inference, and speed is not of the utmost importance,
        the decoding algorithm is NOT parallelized over the sentences in a batch.

        x: A packed sequence representing the following data:
        if batch_first==True:
            x.data has shape batch_size x max_seq_len x num_classes (when padded)
        else:
            x.data has shape max_seq_len x batch_size x num_classes (when padded)

        Returns:
            A python list of tensors, where the ith element
            is the Y_{optimal}^i

        """

        pad_x, batch_sizes=pad_packed_sequence(x, batch_first=self.batch_first)

        if not self.batch_first:
            pad_x=torch.swapaxes(pad_x, 0, 1)

        return [self._viterbi_decode(pad_x[i, :batch_sizes[i], :]) for i in range(pad_x.shape[0])]

    def _viterbi_decode(self, x):

        """
        x: the feature vector of shape (seq_len x self.num_classes). Note how it is
        not max_seq_len.
        Returns the optimal y for this x. So, shape is (seq_len, )
        """

        seq_len, should_be_num_classes=x.shape
        
        assert should_be_num_classes==self.num_classes, "Unexpected and undesirable wreckage."
        
        dp=torch.zeros(seq_len, self.num_classes).to(self.device)
        path=torch.zeros(seq_len, self.num_classes).to(self.device)

        dp[0, :]=self.origination_scores+x[0, :]
        path[0, :]=torch.arange(self.num_classes)
        for i in range(seq_len):
            dp[i, :]=x[i, :]+(self.transition_scores+dp[i-1, :].reshape((-1, 1))).max(axis=1)
            path[i, :]=(self.transition_scores+dp[i-1, :].reshape((-1, 1))).argmax(axis=1)

        res=torch.zeros(seq_len, ).to(self.device).to(torch.uint8)

        res[-1]=dp[-1, :].argmax()
        for i in range(seq_len-2, -1, -1):
            res[i]=path[i+1, res[i+1].item()]

        return res

    def score_tag_sequence(self, x, y):

        """
        Returns a tensor of shape (batch_size, ) where the ith element denotes the probability
        of y[i] being the optimal sequence tagging for x[i]

        x: a packed sequence representing the following data
        if batch_first==True:
            x.data has shape batch_size x max_seq_len x num_classes (when padded)
        else:
            x.data has shape max_seq_len x batch_size x num_classes (when padded)

        y: A packed sequence with the following data
        if batch_first==True:
            y.data has shape batch_size x max_seq_len (when padded)
        else:
            y.data has shape max_seq_len x batch_size (when padded)
        y stores the integer class indices. So elements of y are expected to be 
        in range(self.num_classes)
        """

        pad_x, batch_sizes=pad_packed_sequence(x, batch_first=self.batch_first)
        if not self.batch_first:
            pad_x=torch.swapaxes(pad_x, 0, 1)

        normalizer=self._forward_algo(pad_x, batch_sizes.to(self.device))

        pad_y, y_batch_sizes=pad_packed_sequence(y, batch_first=self.batch_first)
        if not self.batch_first:
            pad_y=torch.swapaxes(pad_y, 0, 1)

        assert (batch_sizes==y_batch_sizes).all()

        # pad_x has shape (N, max_seq_len, self.num_classes)

        # Do not try to make sense of what follows
        pad_y_mod=torch.clone(pad_y)
        pad_y_mod[pad_y==self.target_pad_val]=0

        K=torch.zeros_like(pad_x).to(self.device)
        torch.scatter(K, 2, pad_y_mod[:, :, None], 1)
        emission_val=torch.mul(pad_x, K).sum(dim=(1, 2)) # this has shape (N, )


        transition_matrix=torch.cat((pad_y_mod[:, :-1, None], pad_y_mod[:, 1:, None]), dim=2) # this has shape (N, max_seq_len-1, 2)
        rows, cols, should_be_2=transition_matrix.shape
        assert should_be_2==2, "Unexpected and undesirable wreckage"

        tmp_transition_val=self.transition_scores[transition_matrix.reshape((-1, 2))[:, 0], transition_matrix.reshape((-1, 2))[:, 1]].reshape((rows, cols))
        origination_val=self.origination_scores[pad_y_mod[:, 0]] # has shape (N, )
        transition_val=torch.cat((origination_val.reshape((-1, 1)), tmp_transition_val), 1) # has shape (N, max_seq_len), so sum by reducing on axis 1 to get the numerator

        batch_scores=transition_val.sum(dim=1)

        return (batch_scores-normalizer).sum()

    def _forward_algo(self, pad_x, batch_sizes):

        num_samples, max_seq_len, _=pad_x.shape
        alpha=torch.zeros(max_seq_len, num_samples, self.num_classes).to(self.device)

        alpha[0]=pad_x[:, 0, :]+self.origination_scores.reshape((1, -1)).expand(num_samples, -1)

        # Dont try to make sense of variable names for the next section
        # Dont try to make sense of anything in the next section, infact
        # Life is too short

        for i in range(1, max_seq_len):

            # expand does not allow expanding non-singleton dimensions
            # So, introduce a dummy dimension and then expand
            # Also, expand returns a view, so clone because we need to write to this

            J=torch.clone(
                alpha[i-1][:, None, :].expand(-1, self.num_classes, -1)
                )

            J=J+self.transition_scores

            # J has shape N, M, M 
            alpha[i]=pad_x[:, i, :]+torch.logsumexp(J, axis=2) # Review that reduction is on axis=2

        res=torch.gather(
            torch.swapaxes(alpha, 0, 1),
            1,
            (batch_sizes-1)[:, None, None].expand(num_samples, 1, self.num_classes)
            ).squeeze()

        # res has shape (N, M)

        return res.sum(axis=1) # the returned value has shape (N, )