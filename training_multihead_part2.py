#! /usr/bin/env python3.5

import argparse
import json
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from models import ThreeLayerNet_1LevelAttn_multihead

# NOTE: no validation data set is being used here


# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('model_name', type=str, help = 'name of model to be saved')
commandLineParser.add_argument('--seed', type=int, default = 1, help = 'Specify the global random seed')

args = commandLineParser.parse_args()
seed = args.seed
model_name = args.model_name

# Set seed for reproducibility
torch.manual_seed(seed)

print(seed)
print(model_name)

# force cpu for now
device = torch.device("cpu")


# Get all the data
target_file = 'data4D_training_part2.txt'
with open(target_file, 'r') as f:
	data = json.load(f)

print("Got the data")

# Extract relevant parts of data
X = data[0]
y = data[1]
L_list = data[2]


# Convert to tensors
X = torch.FloatTensor(X)
y = torch.FloatTensor(y)
L = torch.FloatTensor(L_list)

# Replace all zeros in matrix with 1 (so it can be used in variable length RNN)
L[L==0] = 1



X = X.to(device)
y = y.to(device)
L = L.to(device)

# Make the mask from utterance lengths matrix L
M = [[([1]*utt_len + [-100000]*(X.size(2)- utt_len)) for utt_len in speaker] for speaker in L_list]
M = torch.FloatTensor(M)
M = M.to(device)

# Initialise constants
word_dim = 768
h1_dim = 250
h2_dim = 150
y_dim = 1
utt_num = 8

bs = 32
epochs = 45
lr = 8*1e-3

# Store all training dataset in a single wrapped tensor
train_ds = TensorDataset(X, y, M)

# Use DataLoader to handle minibatches easily
train_dl = DataLoader(train_ds, batch_size = bs, shuffle = True)


model = ThreeLayerNet_1LevelAttn_multihead(word_dim, h1_dim, h2_dim, y_dim, utt_num)
model = model.to(device)

print("made model")

criterion = torch.nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(epochs):
	model.train()
	for xb, yb, mb in train_dl:
	
		# Forward pass
		y_pred = model(xb, mb)
	

		# Compute and print loss
		loss = criterion(y_pred[:,0], yb)
	
		# Zero gradients, backward pass, update weights
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	model.eval()
	y_pr = model(X,M)
	mse_loss = criterion(y_pr[:,0], y)
	print(epoch, mse_loss.item())	
	

# Save the model to a file
file_path = 'saved_models/'+model_name+'_seed'+str(seed)+'.pt'
torch.save(model, file_path)




















