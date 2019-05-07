import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
import sys

def tile(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
    return tiled

def apply_attention(input, attention):
    """ Apply any number of attention maps over the input.
        The attention map has to have the same size in all dimensions except dim=1.
    """
    n, c = input.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, c, -1)
    attention = attention.view(n, glimpses, -1)
    s = input.size(2)

    # apply a softmax to each attention map separately
    # since softmax only takes 2d inputs, we have to collapse the first two dimensions together
    # so that each glimpse is normalized separately
    attention = attention.view(n * glimpses, -1)
    attention = F.softmax(attention)

    # apply the weighting by creating a new dim to tile both tensors over
    target_size = [n, glimpses, c, s]
    input = input.view(n, 1, c, s).expand(*target_size)
    attention = attention.view(n, glimpses, 1, s).expand(*target_size)
    weighted = input * attention
    # sum over only the spatial dimension
    weighted_mean = weighted.sum(dim=3)
    # the shape at this point is (n, glimpses, c, 1)
    return weighted_mean.view(n, -1)

class Enc(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(Enc, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        features = None
        with torch.no_grad():
            features = self.resnet(images)
        flat_features = features.reshape(features.size(0), -1)
        flat_features = self.bn(self.linear(flat_features))
        return features, flat_features

DEBUG = False
class Dec(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, ans_vocab_size, num_layers, max_seq_length=26):
        super(Dec, self).__init__()
        self.feature_map_n = 2048 # resnet feature
        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.vocab_size = vocab_size
        self.ans_vocab_size = ans_vocab_size
        self.glimpse_n = 2

        self.language_model = LanguageModel(embed_size, hidden_size, vocab_size, ans_vocab_size, num_layers, max_seq_length=26) 
        self.attention_model = AttentionModel(self.feature_map_n, hidden_size, 64, self.glimpse_n)

        self.classifier_in_n = self.glimpse_n * self.feature_map_n + hidden_size
        self.classifier = Classifier(self.classifier_in_n, 512, ans_vocab_size, drop=0.5)


    def forward(self, raw_features, features, captions, lengths):
        lang_feat = self.language_model(captions, lengths)
        if (DEBUG):
            print 'lang_feat', lang_feat.size()
            print self.attention_model
            print 

        att_lang_feat = self.attention_model(raw_features, lang_feat)
        att_map = apply_attention(raw_features, att_lang_feat)
        if DEBUG:
            print '\nBefore combine'
            print 'att_map', att_map.size()
            print 'lang_feat', lang_feat.size()
        combined = torch.cat([att_map, lang_feat], dim=1)

        if DEBUG:
            print 'combined', combined.size()
            print 'classifier_in_n', self.classifier_in_n

        answer = self.classifier(combined)
        if DEBUG: print 'answer', answer.size()
        return answer


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))

class LanguageModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, ans_vocab_size, num_layers, max_seq_length=26):
        super(LanguageModel, self).__init__()
        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.vocab_size = vocab_size
        self.ans_vocab_size = ans_vocab_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, ans_vocab_size)
        self.max_seq_length = max_seq_length
    
    def forward(self, captions, lengths):
        batch_size = captions.size(0)
        embeddings = self.embed(captions)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 

        hiddens, (hn, cn) = self.lstm(packed)
        # language_output = self.linear(hn[-1])
        # outputs = F.log_softmax(outputs, dim=1)
        # return outputs
        return cn.squeeze(0) # (batch, )


class AttentionModel(nn.Module):
    def __init__(self, img_feat_n, lang_feat_n, filter_1, filter_2):
        drop_rate = 0.0
        super(AttentionModel, self).__init__()
        self.conv1 = nn.Conv2d(img_feat_n, filter_1, 1, bias=False)
        self.fc1 = nn.Linear(lang_feat_n, filter_1)
        self.conv2 = nn.Conv2d(filter_1, filter_2, 1)
        self.drop1 = nn.Dropout(drop_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img_feat, lang_feat):
        if DEBUG: print 'att_img_feat',img_feat.size(),'att_lang_feat',lang_feat.size()
        img_feat = self.conv1(self.drop1(img_feat))
        if DEBUG: print 'img_feat_1', img_feat.size()
        lang_feat = self.fc1(self.drop1(lang_feat))
        if DEBUG: print 'lang_feat_1', img_feat.size()

        # here shape of img_feat == lang_feat (e.g ([20, 64, 1, 1]))
        
        att_feat = tile(lang_feat, img_feat)
        if DEBUG: print 'att_feat_1', att_feat.size()
        
        att_feat = self.relu(att_feat + img_feat)
        if DEBUG: print 'att_feat_2', att_feat.size()

        att_out = self.conv2(self.drop1(att_feat))
        if DEBUG: print 'att_out',att_out.size()
        return att_out

    






