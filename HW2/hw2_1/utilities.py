import torch
import torch.nn as nn
import torch.nn.functional as F
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import DataLoader
import json
import torch.optim as optim
import numpy as np
import tqdm
import os
import numpy as np
import pandas as pd
from collections import Counter
import glob
from tqdm import tqdm
import string
import time
from torch.utils.data import  Dataset
from torch.nn.utils.rnn import pad_sequence

# Preprocess
def create_caption(label_json_path):
    f = open(label_json_path)
    data = json.load(f)
    filename=label_json_path[:-14]+'_newcaption.txt'
    if os.path.exists(filename):
        return filename
    else:
        with open(filename, 'a') as fb:
            fb.write('videoID'+';'+'Caption'+'\n')
            for i in range(len(data)):
                for j in range(len(data[i]['caption'])):
                    fb.write(data[i]['id']+';'+data[i]['caption'][j]+'\n')
        return filename

def process(captions):
    rem_punct = str.maketrans('', '', string.punctuation)
    for i in range(len(captions)):
        line = captions[i]
        line = line.split()

        line = [word.lower() for word in line]

        line = [word.translate(rem_punct) for word in line]

        line = [word for word in line if word.isalpha()]

        captions[i] = ' '.join(line)
    return captions

def numerize(caption, wtoi):
    return [wtoi[word] if word in wtoi else wtoi['<UNK>'] for word in caption.split()]

class Vocabulary:
    def __init__(self, captions, freq_threshold=3):
        self.captions = process(captions)
        self.captions = captions
        self.itow = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.wtoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itow)

    def build_vocab(self,):
        vocab = {}
        idx=4
        for sentence in self.captions:
            for word in sentence.split():
                if word not in vocab:
                    vocab[word] = 1

                else:
                    vocab[word]+=1

                if vocab[word] == self.freq_threshold:
                    self.itow[idx] = word
                    self.wtoi[word] = idx
                    idx+=1

class TrainDataset(Dataset):
    def __init__(self, feat_dir, label_json_path):
        self.df = pd.json_normalize(json.load(open(label_json_path)), meta=['id'], record_path=['caption'])
        self.df.columns = ['caption', 'id']
        self.feat_dir = feat_dir
        self.captions = self.df['caption']
        self.img_feat = self.df['id']
        self.vocab = Vocabulary(self.captions)
        self.vocab.build_vocab()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        feat = self.img_feat[index]

        caption_tokens = caption.split()

        caption_tokens = [word.lower() for word in caption_tokens]

        caption_tokens = [word.translate(str.maketrans('', '', string.punctuation)) for word in caption_tokens]

        caption_tokens = [word for word in caption_tokens if word.isalpha()]

        numericalized_caption = [self.vocab.wtoi["<SOS>"]]
        numericalized_caption += numerize(" ".join(caption_tokens), self.vocab.wtoi)
        numericalized_caption.append(self.vocab.wtoi["<EOS>"])

        return feat, torch.Tensor(np.load(self.feat_dir+feat+'.npy')), torch.tensor(numericalized_caption)

class TestDataset(Dataset):
    def __init__(self, feat_dir):
        self.feat_dir = feat_dir
        self.features = [f[:-4] for f in os.listdir(feat_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        idx = self.features[index]
        feat = os.path.join(self.feat_dir, f'{idx}.npy')
        return idx, torch.Tensor(np.load(feat))

class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        ids = [item[0] for item in batch]
        feats = [item[1].unsqueeze(0) for item in batch]
        feats = torch.cat(feats, dim=0)
        targets = [item[2].clone().detach() for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return ids, feats, torch.transpose(targets, 0, 1)

# Model
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.W1 = nn.Linear(2*hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.W3 = nn.Linear(hidden_size, hidden_size)
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        batch_size, seq_len, feat_n = encoder_outputs.size()
        hidden_state = hidden_state.view(batch_size, 1, feat_n).repeat(1, seq_len, 1)
        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2*self.hidden_size)

        x = self.W1(matching_inputs)
        x = self.W2(x)
        x = self.W3(x)
        attention_weights = self.to_weight(x)
        attention_weights = attention_weights.view(batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.compress = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, input):
        batch_size, seq_len, feat_n = input.size()
        input = input.view(-1, feat_n)
        input = self.compress(input)
        input = self.dropout(input)
        input = input.view(batch_size, seq_len, self.hidden_size)
        output, hidden_state = self.gru(self.dropout(input))

        return output, hidden_state




class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, embed_dim, helper=None, dropout=0.2):
        super(DecoderRNN, self).__init__()
        self.hidden_size, self.vocab_size, self.embed_dim, self.helper = hidden_size, vocab_size, embed_dim, helper

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size+embed_dim, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.linear = nn.Linear(hidden_size,vocab_size)


    def forward(self, encoder_hidden=None, encoder_output=None, targets=None, mode=None, teacher_ratio=0.7):
        batch_size, _, _ = encoder_output.size()
        hidden_state = (self.init_state(batch_size).unsqueeze(0)).to(device)
        seq_logProb = []
        caption_preds = []
        targets = self.embedding(targets)
        _, seq_len, _ = targets.size()

        embed = targets[:, 0]
        for i in range(seq_len-1):
            context = self.attention(hidden_state, encoder_output)
            gru_input = torch.cat([embed, context], dim=1).unsqueeze(1)
            gru_output, hidden_state = self.gru(gru_input, hidden_state)
            logprob = self.linear(self.dropout(gru_output.squeeze(1)))
            seq_logProb.append(logprob.unsqueeze(1))

            use_teacher_forcing = True if random.random() < teacher_ratio else False
            if use_teacher_forcing:
                embed = targets[:, i+1]
            else:
                decoder_input = logprob.unsqueeze(1).max(2)[1]
                embed = self.embedding(decoder_input).squeeze(1)

        seq_logProb = torch.cat(seq_logProb, dim=1)
        caption_preds = seq_logProb.max(2)[1]
        return seq_logProb, caption_preds


    def inference(self, encoder_hidden, encoder_output, vocab):
        batch_size, _, _ = encoder_output.size()
        hidden_state = (self.init_state(batch_size).unsqueeze(0)).to(device)
        decoder_input = torch.tensor(1).view(1,-1).to(device)
        seq_logProb = []
        caption_preds = []
        max_seq_len = 30

        for i in range(max_seq_len-1):
            embed = self.embedding(decoder_input).squeeze(1)
            context = self.attention(hidden_state, encoder_output)
            gru_input = torch.cat([embed, context], dim=1).unsqueeze(1)
            gru_output, hidden_state = self.gru(gru_input, hidden_state)
            logprob = self.linear(gru_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_input = logprob.unsqueeze(1).max(2)[1]

            if vocab.itow[decoder_input.item()] == "<EOS>":
                break

        seq_logProb = torch.cat(seq_logProb, dim=1)
        caption_preds = seq_logProb.max(2)[1]
        return seq_logProb, caption_preds


    def init_state(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size))


class S2VTMODEL(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, embed_dim):
        super(S2VTMODEL, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = DecoderRNN(hidden_size, vocab_size, embed_dim)


    def forward(self, features, target_captions=None, mode=None):
        encoder_outputs, encoder_hidden = self.encoder(features)
        if mode == 'train':
            seq_logProb, caption_preds = self.decoder(encoder_hidden=encoder_hidden, encoder_output=encoder_outputs, targets=target_captions, mode=mode)
        elif mode == 'test':
            seq_logProb, caption_preds = self.decoder.inference(encoder_hidden=encoder_hidden, encoder_output=encoder_outputs, vocab=None)
        else:
            raise KeyError('mode is not valid')
        return seq_logProb, caption_preds