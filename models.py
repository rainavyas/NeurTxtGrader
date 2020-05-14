import torch

class ThreeLayerNet_2LevelAttn(torch.nn.Module):
	def __init__(self, word_dim, h1_dim, h2_dim, y_dim):
		
		super(ThreeLayerNet_2LevelAttn, self).__init__()
		self.attn1 = torch.nn.Linear(word_dim, word_dim, bias = False)
		self.attn2 = torch.nn.Linear(word_dim, word_dim, bias = False)
		self.linear1 = torch.nn.Linear(word_dim, h1_dim)
		self.linear2 = torch.nn.Linear(h1_dim, h2_dim)
		self.linear3 = torch.nn.Linear(h2_dim, y_dim)

		self.word_dim = word_dim
		self.utt_weights = None


	def forward(self, X, mask):
		
		M = mask
		
		# Apply self-attention over the words
		A1 = self.attn1(torch.eye(self.word_dim))
		S_half = torch.einsum('buvi,ij->buvj', X, A1)
		S = torch.einsum('buvi,buvi->buv', X, S_half)
		T = torch.nn.Tanh()
		ST = T(S)
		# Use mask to convert padding scores to -inf (go to zero after softmax normalisation)
		# Note tanh function maintained 0s in ST in correct positions (as uAu^T = 0 when u = 0)
		ST_translated = ST + 1
		ST_masked = ST_translated * M
		# Normalise weights using softmax for each utterance of each speaker
		SM = torch.nn.Softmax(dim = 2)
		W = SM(ST_masked)
		# Perform weighted sum (using normalised scores above) along the words axes for X
		weights_extra_axis = torch.unsqueeze(W, 3)
		repeated_weights = weights_extra_axis.expand(-1, -1, -1, X.size(3))
		x_multiplied = X * repeated_weights
		x_attn1 = torch.sum(x_multiplied, dim = 2)



		# Apply self-attention over the utterances
		A2 = self.attn2(torch.eye(self.word_dim))
		S_half2 = torch.einsum('bui,ij->buj',x_attn1, A2)
		S2 = torch.einsum('bui,bui->bu',x_attn1, S_half2)
		ST2 = T(S2)
		# Normalise using softmax along the utterances dimension
		SM2 = torch.nn.Softmax(dim = 1)
		W2 = SM2(ST2)
		self.utt_weights = W2
		# Perform weight sum (along the utts axes)
		weights_extra_axis2 = torch.unsqueeze(W2, 2)
		repeated_weights2 = weights_extra_axis2.expand(-1, -1, X.size(3))
		x_multiplied2 = x_attn1 * repeated_weights2
		X_final = torch.sum(x_multiplied2, dim = 1)

			 
		
		# Pass through feed-forward DNN
		h1 = self.linear1(X_final).clamp(min=0)
		h2 = self.linear2(h1).clamp(min=0)
		y_pred = self.linear3(h2)
		return y_pred
		


	def get_utt_attn_weights(self):		
		return self.utt_weights








class ThreeLayerNet_1LevelAttn(torch.nn.Module):
	def __init__(self, word_dim, h1_dim, h2_dim, y_dim, utt_num):

		super(ThreeLayerNet_1LevelAttn, self).__init__()
		self.attn1 = torch.nn.Linear(word_dim, word_dim, bias = False)
		self.linear1 = torch.nn.Linear(word_dim*utt_num, h1_dim)
		self.linear2 = torch.nn.Linear(h1_dim, h2_dim)
		self.linear3 = torch.nn.Linear(h2_dim, y_dim)

		self.word_dim = word_dim


	def forward(self, X, mask):

		M = mask

		# Apply self-attention over the words
		A1 = self.attn1(torch.eye(self.word_dim))
		S_half = torch.einsum('buvi,ij->buvj', X, A1)
		S = torch.einsum('buvi,buvi->buv', X, S_half)
		T = torch.nn.Tanh()
		ST = T(S)
		# Use mask to convert padding scores to -inf (go to zero after softmax normalisation)
		# Note tanh function maintained 0s in ST in correct positions (as uAu^T = 0 when u = 0)
		ST_translated = ST + 1
		ST_masked = ST_translated * M
		# Normalise weights using softmax for each utterance of each speaker
		SM = torch.nn.Softmax(dim = 2)
		W = SM(ST_masked)
		# Perform weighted sum (using normalised scores above) along the words axes for X
		weights_extra_axis = torch.unsqueeze(W, 3)
		repeated_weights = weights_extra_axis.expand(-1, -1, -1, X.size(3))
		x_multiplied = X * repeated_weights
		x_attn1 = torch.sum(x_multiplied, dim = 2)



		# Concatenate the utterances
		X_final = x_attn1.view(x_attn1.size(0), -1)


		# Pass through feed-forward DNN
		h1 = self.linear1(X_final).clamp(min=0)
		h2 = self.linear2(h1).clamp(min=0)
		y_pred = self.linear3(h2)
		return y_pred






class ThreeLayerNet_1LevelAttn_RNN(torch.nn.Module):
	def __init__(self, word_dim, RNN_h_dim,  h1_dim, h2_dim, y_dim, utt_num):

		super(ThreeLayerNet_1LevelAttn_RNN, self).__init__()
		self.gru = torch.nn.GRU(input_size = word_dim, hidden_size = RNN_h_dim, num_layers = 1, bias = True, batch_first = True, bidirectional = True)
		self.attn1 = torch.nn.Linear(RNN_h_dim*2, RNN_h_dim*2, bias = False)
		self.linear1 = torch.nn.Linear(RNN_h_dim*2*utt_num, h1_dim)
		self.linear2 = torch.nn.Linear(h1_dim, h2_dim)
		self.linear3 = torch.nn.Linear(h2_dim, y_dim)

		self.RNN_h_dim = RNN_h_dim
		

	def forward(self, X, mask, L):
		
		M = mask

		# Pass all utterance words through RNN
		# Flatten X to eliminate utterances dimension
		Xf = X.view(X.size(0)*X.size(1), X.size(2), X.size(3))
		# Pack padded tensor into a packed sequence object
		lens = L.view(X.size(0)*X.size(1))
		Xp = torch.nn.utils.rnn.pack_padded_sequence(Xf, lens, batch_first = True, enforce_sorted = False)
		# Pass through bidirectional RNN
		outputs, hidden_states = self.gru(Xp)
		# Convert packed object to padded tensor
		O = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first = True, total_length = X.size(2))
		Op = O[0]
		# Unflatten the tensor
		X_after_RNN = torch.reshape(Op, (X.size(0), X.size(1), X.size(2), -1))  	


		# Apply self-attention over the words
		A1 = self.attn1(torch.eye(X_after_RNN.size(3)))
		S_half = torch.einsum('buvi,ij->buvj', X_after_RNN, A1)
		S = torch.einsum('buvi,buvi->buv', X_after_RNN, S_half)
		T = torch.nn.Tanh()
		ST = T(S)
		# Use mask to convert padding scores to -inf (go to zero after softmax normalisation)
		# Note tanh function maintained 0s in ST in correct positions (as uAu^T = 0 when u = 0)
		ST_translated = ST + 1
		ST_masked = ST_translated * M
		# Normalise weights using softmax for each utterance of each speaker
		SM = torch.nn.Softmax(dim = 2)
		W = SM(ST_masked)
		# Perform weighted sum (using normalised scores above) along the words axes for X
		weights_extra_axis = torch.unsqueeze(W, 3)
		repeated_weights = weights_extra_axis.expand(-1, -1, -1, X_after_RNN.size(3))
		x_multiplied = X_after_RNN * repeated_weights
		x_attn1 = torch.sum(x_multiplied, dim = 2)



		# Concatenate the utterances
		X_final = x_attn1.view(x_attn1.size(0), -1)


		# Pass through feed-forward DNN
		h1 = self.linear1(X_final).clamp(min=0)
		h2 = self.linear2(h1).clamp(min=0)
		y_pred = self.linear3(h2)
		return y_pred








class ThreeLayerNet_1LevelAttn_multihead(torch.nn.Module):
	def __init__(self, word_dim, h1_dim, h2_dim, y_dim, utt_num):

		super(ThreeLayerNet_1LevelAttn_multihead, self).__init__()
		self.attn1 = torch.nn.Linear(word_dim, word_dim, bias = False)
		self.attn2 = torch.nn.Linear(word_dim, word_dim, bias = False)
		self.attn3 = torch.nn.Linear(word_dim, word_dim, bias = False)
		self.attn4 = torch.nn.Linear(word_dim, word_dim, bias = False)
		self.linear1 = torch.nn.Linear(word_dim*utt_num*4, h1_dim)
		self.linear2 = torch.nn.Linear(h1_dim, h2_dim)
		self.linear3 = torch.nn.Linear(h2_dim, y_dim)

		self.word_dim = word_dim


	def forward(self, X, mask):

		M = mask

		# Apply 4-head self-attention over the words
		A1 = self.attn1(torch.eye(self.word_dim))
		A2 = self.attn2(torch.eye(self.word_dim))
		A3 = self.attn3(torch.eye(self.word_dim))
		A4 = self.attn4(torch.eye(self.word_dim))
		
		x_attn1 = self._apply_attn(X, M, A1)
		x_attn2 = self._apply_attn(X, M, A2)
		x_attn3 = self._apply_attn(X, M, A3)
		x_attn4 = self._apply_attn(X, M, A4)
		
		# concatentate the mutli-head outputs for each data point and each utterance
		x_headed = torch.cat((x_attn1, x_attn2, x_attn3, x_attn4), dim = 2)

		# Concatenate the utterances
		X_final = x_headed.view(x_headed.size(0), -1)

		# Pass through feed-forward DNN
		h1 = self.linear1(X_final).clamp(min=0)
		h2 = self.linear2(h1).clamp(min=0)
		y_pred = self.linear3(h2)
		return(y_pred)
	
	def _apply_attn(self, X, M, A):
		S_half = torch.einsum('buvi,ij->buvj', X, A)
		S = torch.einsum('buvi,buvi->buv', X, S_half)
		T = torch.nn.Tanh()
		ST = T(S)
		# Use mask to convert padding scores to -inf (go to zero after softmax normalisation)
		# Note tanh function maintained 0s in ST in correct positions (as uAu^T = 0 when u = 0)
		ST_translated = ST + 1
		ST_masked = ST_translated * M
		# Normalise weights using softmax for each utterance of each speaker
		SM = torch.nn.Softmax(dim = 2)
		W = SM(ST_masked)
		# Perform weighted sum (using normalised scores above) along the words axes for X
		weights_extra_axis = torch.unsqueeze(W, 3)
		repeated_weights = weights_extra_axis.expand(-1, -1, -1, X.size(3))
		x_multiplied = X * repeated_weights
		x_attn = torch.sum(x_multiplied, dim = 2)
		return x_attn







class ThreeLayerNet_1LevelAttn_RNN_multihead(torch.nn.Module):
	def __init__(self, word_dim, RNN_h_dim,  h1_dim, h2_dim, y_dim, utt_num):

		super(ThreeLayerNet_1LevelAttn_RNN_multihead, self).__init__()
		self.gru = torch.nn.GRU(input_size = word_dim, hidden_size = RNN_h_dim, num_layers = 1, bias = True, batch_first = True, bidirectional = True)
		self.attn1 = torch.nn.Linear(RNN_h_dim*2, RNN_h_dim*2, bias = False)
		self.attn2 = torch.nn.Linear(RNN_h_dim*2, RNN_h_dim*2, bias = False)
		self.attn3 = torch.nn.Linear(RNN_h_dim*2, RNN_h_dim*2, bias = False)
		self.attn4 = torch.nn.Linear(RNN_h_dim*2, RNN_h_dim*2, bias = False)
		self.linear1 = torch.nn.Linear(RNN_h_dim*2*utt_num*4, h1_dim)
		self.linear2 = torch.nn.Linear(h1_dim, h2_dim)
		self.linear3 = torch.nn.Linear(h2_dim, y_dim)

		self.RNN_h_dim = RNN_h_dim


	def forward(self, X, mask, L):

		M = mask

		# Pass all utterance words through RNN
		# Flatten X to eliminate utterances dimension
		Xf = X.view(X.size(0)*X.size(1), X.size(2), X.size(3))
		# Pack padded tensor into a packed sequence object
		lens = L.view(X.size(0)*X.size(1))
		Xp = torch.nn.utils.rnn.pack_padded_sequence(Xf, lens, batch_first = True, enforce_sorted = False)
		# Pass through bidirectional RNN
		outputs, hidden_states = self.gru(Xp)
		# Convert packed object to padded tensor
		O = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first = True, total_length = X.size(2))
		Op = O[0]
		# Unflatten the tensor
		X_after_RNN = torch.reshape(Op, (X.size(0), X.size(1), X.size(2), -1))


		# Apply 4-head self-attention over the words
		A1 = self.attn1(torch.eye(X_after_RNN.size(3)))
		A2 = self.attn2(torch.eye(X_after_RNN.size(3)))
		A3 = self.attn3(torch.eye(X_after_RNN.size(3)))
		A4 = self.attn4(torch.eye(X_after_RNN.size(3)))

		x_attn1 = self._apply_attn(X_after_RNN, M, A1)
		x_attn2 = self._apply_attn(X_after_RNN, M, A2)
		x_attn3 = self._apply_attn(X_after_RNN, M, A3)
		x_attn4 = self._apply_attn(X_after_RNN, M, A4)

		# concatentate the mutli-head outputs for each data point and each utterance
		x_headed = torch.cat((x_attn1, x_attn2, x_attn3, x_attn4), dim = 2)


		# Concatenate the utterances
		X_final = x_headed.view(x_headed.size(0), -1)


		# Pass through feed-forward DNN
		h1 = self.linear1(X_final).clamp(min=0)
		h2 = self.linear2(h1).clamp(min=0)
		y_pred = self.linear3(h2)
		return y_pred



	def _apply_attn(self, X, M, A):
		S_half = torch.einsum('buvi,ij->buvj', X, A)
		S = torch.einsum('buvi,buvi->buv', X, S_half)
		T = torch.nn.Tanh()
		ST = T(S)
		# Use mask to convert padding scores to -inf (go to zero after softmax normalisation)
		# Note tanh function maintained 0s in ST in correct positions (as uAu^T = 0 when u = 0)
		ST_translated = ST + 1
		ST_masked = ST_translated * M
		# Normalise weights using softmax for each utterance of each speaker
		SM = torch.nn.Softmax(dim = 2)
		W = SM(ST_masked)
		# Perform weighted sum (using normalised scores above) along the words axes for X
		weights_extra_axis = torch.unsqueeze(W, 3)
		repeated_weights = weights_extra_axis.expand(-1, -1, -1, X.size(3))
		x_multiplied = X * repeated_weights
		x_attn = torch.sum(x_multiplied, dim = 2)
		return x_attn
	
