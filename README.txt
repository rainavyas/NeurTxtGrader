MODEL:

ThreeLayerNet_1LevelAttn_multihead located in models.py
This model is trained on 5 sections (A-E), corresponding to different parts of a Linguaskill examination.



DATA:

Linguaskill-Business and Linguaskill-General: these datasets cannot be made available
Description of data given in paper



PREPROCESSING:

Remove hesitation tokens from ASR transcript.
Remove silence tokens from ASR transcript.
Use BERT (from huggingface website) to produce contextual word embeddings (786 dim vectors)
Create mask indicating length of each response, when feeding into model

The preprocessing steps have been clearly outlined in greedy_word_search.py
This file also outlines a simple discrete greedy search approach for text based universal adversarial attacks



TRAINING:

Each section has a separate training algorithm - files provided in directory
Time complexity ~ 2 hours for 900 speakers (section C for example)
hyperparameters - specifc model hyperparameters given in training files
Repeated for 10 different seeds to generate an ensemble of models


Can be generally achieved using TrainNeurTxtGrader.py


EVALUATION:

~ 200 speakers
Ensemble predictions simply averaged
Statistics calculated: pcc, rmse, <0.5, <1.0 (refer to paper)

Can be generally achieved using EvalNeurTxtGrader.py


DEPENDENCIES:

pytorch (any version)
transformer package -> required for BERT transformation



COMPUTING:

Training can be performed on GPU or CPU (training files to be adjusted as required)

