import torch
import torch.nn as nn
import torchvision.models as models
from naive import Enc
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data
import pickle
from utils import img_data_2_mini_batch
from torchvision import transforms


# ==========================================================================

val_data_json = json.load(open('cocoqa_data_prepro_93.json', 'r'))

itow = val_data_json['ix_to_word']
itoa = val_data_json['ix_to_ans']
unique_img_val = val_data_json['unique_img_val']

# ==========================================================================

val_data_h5 = h5py.File('cocoqa_data_prepro_93.h5', 'r')

ques_val = val_data_h5['ques_val'][:]
ans_val = val_data_h5['ans_val'][:]
question_id_val = val_data_h5['question_id_val'][:]
img_pos_val = val_data_h5['img_pos_val'][:]
ques_length_val = val_data_h5['ques_length_val'][:]

# ==========================================================================

img_data = pickle.load(open('img_data_19.pkl', 'rb'))

# ==========================================================================

img_pos_val = torch.from_numpy(img_pos_val)
ques_val = torch.from_numpy(ques_val)
ans_val = torch.from_numpy(ans_val)

print(img_pos_val.shape)
print(ques_val.shape)
print(ans_val.shape)

train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(img_pos_val, ques_val, ans_val),
        batch_size=100,
        shuffle=True,
        num_workers=2
    )

encoder = Enc(embed_size=128)
encoder.double()

transform = transforms.Compose([
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

# start your train
for epoch in range(10):
    for i, (img_pos, ques, ans) in enumerate(train_loader):

        img_mini_batch = img_data_2_mini_batch(img_pos, img_data, 100)
        img_mini_batch = transform(img_mini_batch)
        print(img_mini_batch.shape)

        # Do LSTM for the questions -> embedded_words
        # embedded_words = embedded_words + features
        # Do LSTM
        # calculate loss with labels
        # keep training
