import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class EncDec(nn.Module):
    def __init__(self, embed_size, hidden_size, attention_size, vocab_size, ans_vocab_size, num_layers, debug):
        super(EncDec, self).__init__()

        self.encoder = Enc(debug)
        self.decoder = Dec(embed_size, hidden_size, attention_size, vocab_size, ans_vocab_size, num_layers, debug)

    def forward(self, images, questions, lengths):
        img_features = self.encoder(images)
        logits = self.decoder(img_features, questions, lengths)

        logits = F.log_softmax(logits, dim=1)

        return logits

    def get_parameters(self):
        params = list(self.decoder.parameters()) + \
                 list(self.encoder.linear.parameters()) + \
                 list(self.encoder.bn.parameters())
        return params


class Enc(nn.Module):
    def __init__(self, debug):
        super(Enc, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.debug = debug

    def forward(self, images):
        with torch.no_grad():
            features_maps = self.resnet(images)

        if self.debug:
            print('\ndone image encoder:')
            print('features_maps:', features_maps.size())

        return features_maps


class Dec(nn.Module):
    def __init__(self, embedding_features, lstm_features, attention_features, ques_vocab_size, ans_vocab_size, lstm_layers, debug):
        super(Dec, self).__init__()
        self.image_features = 2048
        self.glimpses = 2
        self.debug = debug

        self.question_encoder = QuestionEncoder(ques_vocab_size, embedding_features, lstm_features, lstm_layers, debug, drop=0.5)
        self.attention_model = Attention(self.image_features,
                                         lstm_features,
                                         attention_features,
                                         self.glimpses,
                                         debug=debug,
                                         drop=0.5,)

        self.classifier = Classifier(self.glimpses * self.image_features + lstm_features, lstm_features, ans_vocab_size, drop=0.5)

    def forward(self, features_maps, questions, lengths):

        question_features = self.question_encoder(questions, lengths)

        attention_features = self.attention_model(features_maps, question_features)

        attn_features_maps = apply_attention(features_maps, attention_features, self.debug)

        combined = torch.cat([attn_features_maps, question_features], dim=1)

        answer = self.classifier(combined)

        if self.debug:
            print('\ndone cat and fc:')
            print('combined features:', combined.size())
            print('answer:', answer.size())

        return answer


class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, debug, drop):
        super(QuestionEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.debug = debug

    def forward(self, questions, lengths):
        embedded = self.embed(questions)
        tanhed = self.tanh(self.drop(embedded))
        packed = pack_padded_sequence(tanhed, lengths, batch_first=True)
        hiddens, (hn, cn) = self.lstm(packed)

        if self.debug:
            print('\ndone questions encoder:')
            print('question features:', cn.squeeze(0).size())

        return cn.squeeze(0)


class Attention(nn.Module):
    def __init__(self, features_maps, question_features, attention_features, glimpses, debug, drop):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(features_maps, attention_features, 1, bias=False)
        self.q_linear = nn.Linear(question_features, attention_features)
        self.glimpses_conv = nn.Conv2d(attention_features, glimpses, 1)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        self.debug = debug

    def forward(self, v, q):

        # image features to attention features
        v = self.v_conv(self.drop(v))
        # print(v.size())
        # question features to attention features
        q = self.q_linear(self.drop(q))
        # print(q.size())
        # tile question features to feature maps
        tiled_q = tile_2d(v, q)
        # print(tiled_q.size())
        # combine v and q
        vq = self.relu(v + tiled_q)

        glimpses = self.glimpses_conv(self.drop(vq))

        if self.debug:
            print('\ndone attention:')
            print('image_features:', v.size())
            print('question_features:', q.size())
            print('tiled_features:', tiled_q.size())
            print('combined vq:', vq.size())
            print('glimpses:', glimpses.size())

        return glimpses


def apply_attention(features_maps, attention, debug):
    # print('\ndone apply attention:')
    # print(features_maps.size())
    # print(attention.size())
    n, c = features_maps.size()[:2]
    glimpses = attention.size(1)
    # print('n, c, glimpses:', n, c, glimpses)
    features_maps = features_maps.view(n, 1, c, -1)
    # print('features maps:', features_maps.size())
    attention = attention.view(n, glimpses, -1)
    # print('attention:', attention.size())
    attention = F.softmax(attention, dim=-1)
    # print('attention:', attention.size())
    attention = attention.unsqueeze(2) 
    # print('attention:', attention.size())
    weighted = attention * features_maps 
    # print('weighted:', weighted.size())
    weighted_mean = weighted.sum(dim=-1) 
    # print('weighted mean:', weighted_mean.size())
    # print('weighted mean return:', weighted_mean.view(n, -1).size())

    if debug:
        print('\ndone apply attention:')
        print('n, c, glimpses:', n, c, glimpses)
        print('features maps:', features_maps.size())
        print('attention:', attention.size())
        print('weighted:', weighted.size())
        print('weighted mean:', weighted_mean.size())
        print('weighted mean return:', weighted_mean.view(n, -1).size())

    return weighted_mean.view(n, -1)


def tile_2d(feature_maps, feature_vec):
    n, c = feature_vec.size()
    spatial_size = feature_maps.dim() - 2
    tiled = feature_vec.view(n, c, *([1] * spatial_size)).expand_as(feature_maps)
    return tiled


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))
