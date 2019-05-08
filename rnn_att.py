import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class Enc(nn.Module):
    def __init__(self, embed_size):
        super(Enc, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        features = None
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class Dec(nn.Module):
    def __init__(self, embed_size, 
                       hidden_size, 
                       vocab_size, 
                       ans_vocab_size, 
                       num_layers, 
                       max_seq_length=26,
                       rnn_type='lstm',
                       use_prefix_n=2):
        super(Dec, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        else: 
            self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)

        self.prefix_n = max(1, use_prefix_n)
        n_prefix_hidden_n = 1 + self.prefix_n
        self.linear = nn.Linear(n_prefix_hidden_n*hidden_size, ans_vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        batch_size = captions.size(0)
        embeddings = self.embed(captions)
        unsq = features.unsqueeze(1)

        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        # print 'packed',packed

        hiddens, (hn, cn) = self.rnn(packed)
        tmp = torch.nn.utils.rnn.pad_packed_sequence(hiddens)

        hiddens, hidden_lengths = tmp
        print(hiddens.shape)
        hn_l = [hiddens[i] for i in range(self.prefix_n+1)] + [hn[-1]]
        outputs = torch.cat(hn_l, dim=1)
        print('output shape', outputs.shape)
        outputs = self.linear(outputs)
        outputs = F.log_softmax(outputs, dim=1)
        return outputs

class EncDec(nn.Module):
    def __init__(self, embed_size, 
                       hidden_size, 
                       vocab_size, 
                       ans_vocab_size, 
                       num_layers, 
                       max_seq_length=26,
                       rnn_type='lstm',
                       prefix_n=1):

        super(EncDec, self).__init__()
        self.embed_size     = embed_size
        self.hidden_size    = hidden_size
        self.vocab_size     = vocab_size
        self.ans_vocab_size = ans_vocab_size
        self.num_layers     = num_layers
        self.max_seq_length = max_seq_length

        self.encoder = Enc(embed_size)
        self.decoder = Dec(embed_size, hidden_size, vocab_size, ans_vocab_size, 
                           num_layers, max_seq_length, rnn_type, use_prefix_n=prefix_n)
    
    def forward(self, images, questions, lengths):
        img_features = self.encoder(images)
        logits = self.decoder(img_features, questions, lengths)
        return logits 
  
    def get_parameters(self):
        params = list(self.decoder.parameters()) +\
                 list(self.encoder.linear.parameters()) +\
                 list(self.encoder.bn.parameters())
        return params
