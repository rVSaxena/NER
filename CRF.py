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

		return

	def decode(self, x):

		"""
		Because decode is used for inference, and speed is not of the utmost importance,
		the decoding algorithm is NOT parallelized over the sentences in a batch.

		x: A packed padded sequence representing the following data:
		if batch_first==True:
			x.data has shape batch_size x max_seq_len x num_classes
		else:
			x.data has shape max_seq_len x batch_size x num_classes

		Returns:
			A python list of tensors, where the ith element
			is the Y_{optimal}^i

		"""

		pad_x, batch_sizes=pad_packed_sequence(x, batch_first=self.batch_first)

		if not self.batch_first:
			pad_x=torch.swapaxes(pad_x, 0, 1)

