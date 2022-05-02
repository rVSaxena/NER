import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class CRF(nn.Module):

	"""
	Implements linear chain CRFs. Vectorized operations
	keep it fast. 
	"""

	# This implementation does not use an <end> tag.
	# Can be changed to use, but no use yet.
	

	def __init__(self, num_classes, batch_first):
		
		self.transition_scores=nn.parameter.Parameter(data=torch.rand(num_classes, num_classes))
		self.origination_scores=nn.parameter.Parameter(data=torch.rand(num_classes, ))
		self.batch_first=batch_first
		self.num_classes=num_classes

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

		return self._viterbi_decode(pad_x, batch_sizes)

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
		y stores the integer class indices.
		"""

		pad_x, batch_sizes=pad_packed_sequence(x, batch_first=self.batch_first)
		if not self.batch_first:
			pad_x=torch.swapaxes(pad_x, 0, 1)

		normalizer=self._forward_algo(pad_x, batch_sizes)

		# TODO implement numerator

	def _forward_algo(self, pad_x, batch_sizes):

		num_samples, max_seq_len, _=pad_x.shape
		alpha=torch.zeros(max_seq_len, num_samples, self.num_classes)

		alpha[0]=pad_x[:, 0, :]+self.origination_scores.reshape((1, -1)).expand(num_samples, -1)

		# Dont try to make sense of variable names for the next section
		# Life is too short
		# Dont try to make sense of anything in the next section, infact

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
			



