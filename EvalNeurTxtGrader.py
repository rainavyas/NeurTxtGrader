import argparse
import json
import torch
from transformers import *
from models import ThreeLayerNet_1LevelAttn_multihead

# Get all arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('--trained_model', type = str, help = 'Location of trained .pt model')
commandLineParser.add_argument('--input_section_file', type = str, help = 'Location of data input text file')
commandLineParser.add_argument('--num_utts', type = int, help = 'Specify the number of utterances in this section')

args = commandLineParser.parse_args()

trained_model = args.trained_model
data_file = args.input_section_file
num_utts = args.num_utts

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


# Create separate list of ids and speaker utterances
ids = []
vals = []

for id in utt_embs:
    ids.append(id)
    vals.append(utt_embs[id])


# Convert to appropriate 4D tensor

# Initialise list to hold all input data in tensor format
X = []
remaining_ids = []

# Initialise 2D matrix format to store all utterance lengths per speaker
utt_lengths_matrix = []

for utts, id in zip(vals, ids):
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
	remaining_ids.append(id)
	utt_lengths_matrix.append(utt_lengths)

# Convert to tensors
X = torch.FloatTensor(X)
L = torch.FloatTensor(utt_lengths_matrix)

# Make the mask from utterance lengths matrix L
M = [[([1]*utt_len + [-100000]*(X.size(2)- utt_len)) for utt_len in speaker] for speaker in utt_lengths_matrix]
M = torch.FloatTensor(M)



# force cpu for now
device = torch.device("cpu")
X = X.to(device)
L = L.to(device)
M = M.to(device)

# Load up the model
model = torch.load(trained_model)
model.eval()

# Pass through model to get the predictions
y = model(X, M)
y = y[:,0]
y[y>6] = 6
y[y<0] = 0

y_list = y.tolist()

# Write the speaker ids and predictions to a results file
out_file = 'results.txt'
f =  open(out_file, "w+")

for id, pred in zip(remaining_ids, y_list):
	f.write(id, pred)

f.close()
