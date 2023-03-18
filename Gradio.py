import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import stopwords 
import nltk
from collections import Counter
import string
import re
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle
import Gradio as gr

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

#model class

class SentimentLSTM(nn.Module):
    def __init__(self, no_layers, vocab_size, hidden_dim, embedding_dim, drop_prob = 0.5):
        super(SentimentLSTM, self).__init__()
        
        self.no_layers = no_layers
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        #embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  
        
        #LSTM
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = self.hidden_dim, num_layers = no_layers, batch_first=True)
        
        #dropout layers
        self.dropout = nn.Dropout(0.3)
        
        #linear and Sigmoid layer
        
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()
        
    def forward(self, x, hidden):
        # we just passed a batch
        batch_size = x.size(0) # batch size -> B
        #embed shape -> [B, max_len, embed_dim]
        embeds = self.embedding(x)
        
        
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        
        # drop out and fully connected
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        # sigmoid 
        
        sig_out = self.sig(out)
        
        #reshape to batch size first
        
        sig_out = sig_out.view(batch_size, -1)
        
        sig_out = sig_out[:, -1]
        
        
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        
        # create hidden state and cell state tensors with size [no_layers x batch_size x hidden_dim]
        
        hidden_state = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        cell_state = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        hidden = (hidden_state, cell_state)
        return hidden

# import saved model with weights from pickle file


model = pickle.load(open('model.pkl', 'rb'))
vocab = pickle.load(open('vocab.pkl', 'rb'))


# pre-processing input data

def preprocess_string(s):
    # remove all characters except letters and digits
    s = re.sub(r"[^\w\s]", '', s)
    #remove all extra whites spaces
    s = re.sub(r"\s+", '', s)
    #remove digits
    s = re.sub(r"\d", '', s)
    return s

def padding(sents, seq_len):
    features = np.zeros((len(sents), seq_len), dtype = int)
    for i, rev in enumerate(sents):
        if len(rev) != 0:
            features[i, -len(rev):] = np.array(rev)[:seq_len]
    return features



# predict sentiment of given text


def predict_sentiment(text):
    word_seq = np.array([vocab[preprocess_string(word)] for word in text.split() if preprocess_string(word) in vocab.keys()])
    word_seq = np.expand_dims(word_seq, axis = 0)
    # print(word_seq)
    pad = torch.from_numpy(padding(word_seq, 500))
    
    inputs = pad.to(device)
    batch_size = 1
    h = model.init_hidden(batch_size)
    output, h = model(inputs, h)
    prob = output.item()
    pred = ''
    if prob > 0.5:
        pred = f"This Statement is Positive, with probability of {pred}"
    else:
        pred = f"This Statement is Nagative, with probability of {pred}"
    return pred


# GRadio UI

with gr.Blocks() as demo:
   passage = gr.Textbox(label='Passage')
   submit_btn = gr.Button('Submit')
   label = gr.Label()
   submit_btn.click(predict_sentiment, passage, label)



demo.launch(server_port = 8080)


