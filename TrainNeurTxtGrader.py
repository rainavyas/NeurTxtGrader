import argparse
import json
import torch
from transformers import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from models import ThreeLayerNet_1LevelAttn_multihead

# Get all arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('--input_section_file', type = str, help = 'Location of data input text file')
commandLineParser.add_argument('--grades_file', type = str, help = 'Location of expert/operational grades for all sections')
commandLineParser.add_argument('--output_file', type = str, help = 'specify where to save the trained model')
commandLineParser.add_argument('--section_num', type = int, help = 'Section number, e.g. A = 1, B = 2 etc.. E = 5')
commandLineParser.add_argument('--num_utts', type = int, help = 'Specify the number of utterances in this section')
commandLineParser.add_argument('--seed', type = int, default = 1, help = 'Specify the seed')

args = commandLineParser.parse_args()
data_file = args.input_section_file
grades_file = args.grades_file
output_file = args.output_file
section_num = args.section_num
num_utts = args.num_utts
seed = args.seed

MAX_UTTS_PER_SPEAKER_PART = num_utts
MAX_WORDS_IN_UTT = 200

# Load useful data
with open(data_file, 'r') as f:
	utterances = json.loads(f.read())

print("Loaded Data")

# Convert json output from unicode to string
utterances = [[str(item[0]), str(item[1])] for item in utterances]


# Load tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_basic_tokenize=False, do_lower_case=True)
bert_model = BertModel.from_pretrained('bert-base-cased')
bert_model.eval()

print("Loaded BERT model")



# Convert sentences to a list of BERT embeddings (embeddings per word)
# Store as dict of speaker id to utterances list (each utterance a list of embeddings)
utt_embs = {}
for item in utterances:
	fileName = item[0]
	speakerid = fileName[:12]
	sentence = item[1]

	tokenized_text = tokenizer.tokenize(sentence)
	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
	if len(indexed_tokens) < 1:
		word_vecs = [[0]*768]
	else:
		tokens_tensor = torch.tensor([indexed_tokens])
		with torch.no_grad():
			encoded_layers, _ = bert_model(tokens_tensor)
		bert_embs = encoded_layers.squeeze(0)
		word_vecs = bert_embs.tolist()
	if speakerid not in utt_embs:
		utt_embs[speakerid] =  [word_vecs]
	else:
		utt_embs[speakerid].append(word_vecs)


# get speakerid to section grade dict
grade_dict = {}

lines = [line.rstrip('\n') for line in open(grades_file)]
for line in lines[1:]:
        speaker_id = line[:12]
        grade_overall = line[-3:]
        grade1 = line[-23:-20]
        grade2 = line[-19:-16]
        grade3 = line[-15:-12]
        grade4 = line[-11:-8]
        grade5 = line[-7:-4]
        grades = [grade1, grade2, grade3, grade4, grade5, grade_overall]

        grade = float(grades[section_num-1])
        grade_dict[speaker_id] = grade


# Create list of grades and speaker utterance in same speaker order
grades = []
vals = []

for id in grade_dict:
    grades.append(grade_dict[id])
    vals.append(utt_embs[id])


# Convert to appropriate 4D tensor

# Initialise list to hold all input data in tensor format
X = []
y = []


# Initialise 2D matrix format to store all utterance lengths per speaker
utt_lengths_matrix = []

for utts, grade in zip(vals, grades):
	new_utts = []

	# Reject speakers with not exactly correct number of utterances in part
	if len(utts) != MAX_UTTS_PER_SPEAKER_PART:
		continue


	# Create list to store utterance lengths
	utt_lengths = []

	for curr_utt in utts:
		num_words = len(curr_utt)

		if num_words <= MAX_WORDS_IN_UTT:
			# append padding of zero vectors
			words_to_add = MAX_WORDS_IN_UTT - num_words
			zero_vec_word = [0]*768
			zero_vec_words = [zero_vec_word]*words_to_add
			new_utt = curr_utt + zero_vec_words
			utt_lengths.append(num_words)
		else:
			# Shorten utterance from end
			new_utt = curr_utt[:MAX_WORDS_IN_UTT]
			utt_lengths.append(MAX_WORDS_IN_UTT)

		# Convert all values to float
		new_utt = [[float(i) for i in word] for word in new_utt]

		new_utts.append(new_utt)

	X.append(new_utts)
	y.append(grade)
	utt_lengths_matrix.append(utt_lengths)

# Convert to tensors
X = torch.FloatTensor(X)
y = torch.FloatTensor(y)
L = torch.FloatTensor(utt_lengths_matrix)

# Make the mask from utterance lengths matrix L
M = [[([1]*utt_len + [-100000]*(X.size(2)- utt_len)) for utt_len in speaker] for speaker in utt_lengths_matrix]
M = torch.FloatTensor(M)


# Set seed for reproducibility
torch.manual_seed(seed)

# force cpu for now
device = torch.device("cpu")
X = X.to(device)
y = y.to(device)
L = L.to(device)
M = M.to(device)


# Declare hyperparameters
word_dim = 768
h1_dim = 250
h2_dim = 150
y_dim = 1
utt_num = num_utts

bs = 32
epochs = 45
lr = 0.007

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
torch.save(model, output_file)
