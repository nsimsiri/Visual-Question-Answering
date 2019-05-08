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

        self.question_encoder = QuestionEncoder(ques_vocab_size, embedding_features, lstm_features, lstm_layers, debug)
        self.attention_model = Attention(self.image_features,
                                         lstm_features,
                                         attention_features,
                                         self.glimpses,
                                         debug=debug)

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
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, debug, drop=0.0):
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
    def __init__(self, features_maps, question_features, attention_features, glimpses, debug, drop=0.0):
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
    features_maps = features_maps.view(n, 1, c, -1) # [n, 1, c, s]
    # print('features maps:', features_maps.size())
    attention = attention.view(n, glimpses, -1)
    # print('attention:', attention.size())
    attention = F.softmax(attention, dim=-1)
    # print('attention:', attention.size())
    attention = attention.unsqueeze(2) # [n, g, 1, s]
    # print('attention:', attention.size())
    weighted = attention * features_maps # [n, g, v, s]
    # print('weighted:', weighted.size())
    weighted_mean = weighted.sum(dim=-1) # [n, g, v]
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
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn.init as init
# from torch.nn.utils.rnn import pack_padded_sequence
#
# import torchvision.models as models
#
#
# class Enc(nn.Module):
#     def __init__(self):
#         """Load the pretrained ResNet-152 and replace top fc layer."""
#         super(Enc, self).__init__()
#         resnet = models.resnet152(pretrained=True)
#         modules = list(resnet.children())[:-1]  # delete the last fc layer.
#         self.resnet = nn.Sequential(*modules)
#         # self.linear = nn.Linear(resnet.fc.in_features, embed_size)
#         # self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
#
#     def forward(self, images):
#         with torch.no_grad():
#             features = self.resnet(images)
#         # features = features.reshape(features.size(0), -1)
#         # features = self.bn(self.linear(features))
#         return features
#
#
# class Net(nn.Module):
#     """ Re-implementation of ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]
#     [0]: https://arxiv.org/abs/1704.03162
#     """
#
#     def __init__(self, embedding_tokens, ans_vocab_size):
#         super(Net, self).__init__()
#         question_features = 1024
#         vision_features = 2048
#         glimpses = 2
#
#         self.text = TextProcessor(
#             embedding_tokens=embedding_tokens,
#             embedding_features=300,
#             lstm_features=question_features,
#             drop=0.5,
#         )
#         self.attention = Attention(
#             v_features=vision_features,
#             q_features=question_features,
#             mid_features=512,
#             glimpses=2,
#             drop=0.5,
#         )
#         self.classifier = Classifier(
#             in_features=glimpses * vision_features + question_features,
#             mid_features=1024,
#             out_features=ans_vocab_size,
#             drop=0.5,
#         )
#
#         # for m in self.modules():
#         #     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
#         #         init.xavier_uniform(m.weight)
#         #         if m.bias is not None:
#         #             m.bias.data.zero_()
#
#     def forward(self, v, q, q_len):
#         q = self.text(q, list(q_len.data))
#
#         print(q.size())
#
#         # v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
#         a = self.attention(v, q)
#         v = apply_attention(v, a)
#
#         combined = torch.cat([v, q], dim=1)
#         answer = self.classifier(combined)
#         return answer
#
#
# class Classifier(nn.Sequential):
#     def __init__(self, in_features, mid_features, out_features, drop=0.0):
#         super(Classifier, self).__init__()
#         self.add_module('drop1', nn.Dropout(drop))
#         self.add_module('lin1', nn.Linear(in_features, mid_features))
#         self.add_module('relu', nn.ReLU())
#         self.add_module('drop2', nn.Dropout(drop))
#         self.add_module('lin2', nn.Linear(mid_features, out_features))
#
#
# class TextProcessor(nn.Module):
#     def __init__(self, embedding_tokens, embedding_features, lstm_features, drop=0.0):
#         super(TextProcessor, self).__init__()
#         self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
#         self.drop = nn.Dropout(drop)
#         self.tanh = nn.Tanh()
#         self.lstm = nn.LSTM(input_size=embedding_features,
#                             hidden_size=lstm_features,
#                             num_layers=1)
#         self.features = lstm_features
#
#         self._init_lstm(self.lstm.weight_ih_l0)
#         self._init_lstm(self.lstm.weight_hh_l0)
#         self.lstm.bias_ih_l0.data.zero_()
#         self.lstm.bias_hh_l0.data.zero_()
#
#         init.xavier_uniform(self.embedding.weight)
#
#     def _init_lstm(self, weight):
#         for w in weight.chunk(4, 0):
#             init.xavier_uniform(w)
#
#     def forward(self, q, q_len):
#         embedded = self.embedding(q)
#         tanhed = self.tanh(self.drop(embedded))
#         packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
#         _, (_, c) = self.lstm(packed)
#         return c.squeeze(0)
#
#
# class Attention(nn.Module):
#     def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
#         super(Attention, self).__init__()
#         self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
#         self.q_lin = nn.Linear(q_features, mid_features)
#         self.x_conv = nn.Conv2d(mid_features, glimpses, 1)
#
#         self.drop = nn.Dropout(drop)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, v, q):
#         v = self.v_conv(self.drop(v))
#         q = self.q_lin(self.drop(q))
#         q = tile_2d_over_nd(q, v)
#         x = self.relu(v + q)
#         x = self.x_conv(self.drop(x))
#         return x
#
#
# def apply_attention(input, attention):
#     """ Apply any number of attention maps over the input. """
#     n, c = input.size()[:2]
#     glimpses = attention.size(1)
#
#     # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
#     input = input.view(n, 1, c, -1) # [n, 1, c, s]
#     attention = attention.view(n, glimpses, -1)
#     attention = F.softmax(attention, dim=-1).unsqueeze(2) # [n, g, 1, s]
#     weighted = attention * input # [n, g, v, s]
#     weighted_mean = weighted.sum(dim=-1) # [n, g, v]
#     return weighted_mean.view(n, -1)
#
#
# def tile_2d_over_nd(feature_vector, feature_map):
#     """ Repeat the same feature vector over all spatial positions of a given feature map.
#         The feature vector should have the same batch size and number of features as the feature map.
#     """
#     n, c = feature_vector.size()
#     spatial_size = feature_map.dim() - 2
#     tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
#     return tiled
