import argparse
import json
import torch
from transformers import *
from models import ThreeLayerNet_1LevelAttn_RNN

# n-1 other words to add - add space before each word added to string
other_words = 'offensively obese astronauts amazingly criticizes amazingly psychologically libel earmarks amazingly exercising offensively offensively miraculously over offensively astronauts quite proving amazingly by arguably'
k = 23

# Get the part and its seed to adversarially attack
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('--part', type = int, default = 3, help = 'Specify the part to attack')
commandLineParser.add_argument('--seed', type = int, default = 1, help = 'Specify the seed of particular trained part model')

args = commandLineParser.parse_args()
part = args.part
seed = args.seed

print("part: " + str(part))
print("seed: " + str(seed))


# Define the number of utterances per speaker per part
utt_part_vals = [6, 8, 1, 1, 5]
MAX_UTTS_PER_SPEAKER_PART = utt_part_vals[part-1]

# Define threshold for max number of words per utterance
MAX_WORDS_IN_UTT = 200

trained_model_to_load = 'part'+str(part)+'_multihead_seed'+str(seed)+'.pt'

#Load the model to attack
model_path = '../../../saved_bert_models/'+trained_model_to_load
model = torch.load(model_path)
model.eval()


# Load the relevant part 'useful' data
data_file = '../data/BLXXXgrd02_200/useful_part'+str(part)+'.txt'
with open(data_file, 'r') as f:
	utterances = json.loads(f.read())

print("Loaded Data")

# Convert json output from unicode to string
utterances = [[str(item[0]), str(item[1])] for item in utterances]

# Add the n-1 other words to every utterance
utterances = [[item[0], item[1]+' '+other_words] for item in utterances]

# Create list of words to iterate through for appending
#words_file = 'word2vec/best_wordsk1.txt'
words_file = '../word2vec/best_words/k1.txt'
with open(words_file, 'r') as f:
	test_words = json.loads(f.read())
test_words = [str(word[0]).lower() for word in test_words]

# Add blank word at beginning of list
test_words = ['']+test_words

print("words to check: "+ str(len(test_words)))

# Load tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_basic_tokenize=False, do_lower_case=True)
bert_model = BertModel.from_pretrained('bert-base-cased')
bert_model.eval()

print("Loaded BERT model")


#Define threshold to beat
best = ['none', 0]

# Continue an algorithm from different point
counter = 0

for new_word in test_words:

	# Add new_word to every utterance
	new_utterances = [[item[0], item[1]+' '+new_word] for item in utterances]

	# Convert sentences to a list of BERT embeddings (embeddings per word)
	# Store as dict of speaker id to utterances list (each utterance a list of embeddings)
	utt_embs = {}
	for item in new_utterances:
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

	
	# Convert to appropriate 4D tensor
			
	vals = list(utt_embs.values())

	# Initialise list to hold all input data in tensor format
	X = []

	# Initialise 2D matrix format to store all utterance lengths per speaker
	utt_lengths_matrix = []

	for utts in vals:
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
		utt_lengths_matrix.append(utt_lengths)

	# Convert to tensors
	X = torch.FloatTensor(X)
	L = torch.FloatTensor(utt_lengths_matrix)

	# Make the mask from utterance lengths matrix L
	M = [[([1]*utt_len + [-100000]*(X.size(2)- utt_len)) for utt_len in speaker] for speaker in utt_lengths_matrix]
	M = torch.FloatTensor(M)


	# Pass through the trained model to get y_predictions
	y_adv = model(X, M)

	y_adv = y_adv[:, 0]
	y_adv[y_adv>6] = 6
	y_adv[y_adv<0] = 0

	avg = torch.mean(y_adv)

	if avg > best[1]:
		best = [new_word, avg]
		print(new_word, avg)
	counter += 1
	print(counter)
 
# Write best word to file

best = [best[0], best[1].item()]

file_name = 'thewords/k'+str(k)+ '.txt'
with open(file_name, 'w') as f:
	f.truncate(0)
	f.write(json.dumps(best))

