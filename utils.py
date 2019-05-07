import numpy as np
from PIL import Image
import os
import h5py
import pickle
import json
import torch
import sys
import matplotlib.pyplot as plt
from torchvision import transforms

default_transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))])

def img_data_2_mini_batch(pos_mini_batch, img_data, batch_size):
    pos_mini_batch = pos_mini_batch.numpy()
    img_mini_batch = np.zeros((batch_size, 3, 256, 256))
    for i, pos in enumerate(pos_mini_batch):
        img_mini_batch[i, :, :, :] = img_data[pos]

    return img_mini_batch


# def imgs2batch(img_names, img_positions):
#     img_data = {}
#     for pos in img_positions:
#         img = imread('data/' + img_names[pos])
#         img = np.transpose(img, (2, 0, 1))
#         if pos not in img_data.keys():
#             img_data[pos] = img

#     return img_data

def imgs2batch(img_names, img_positions, transform=default_transform):
    img_data = []
    for pos in img_positions:
        img = imread('data/' + img_names[pos], transform=transform)
        # if (transform is None):
        #     img = np.transpose(img, (2, 0, 1))
        img_data.append(img)
    return img_data



def imread(path, transform=default_transform):
    if not os.path.exists(path):
        raise Exception("IMG_LOAD_ERR - Image File idx={}: [{}] not found".format(idx, img_path))
    img = Image.open(path)
    img = img.resize((256, 256))
    if (transform is not None):
        img = transform(img)

    img = np.array(img)#, dtype=float)
    return img


def gray2rgb(img):
    h, w = img.shape
    rgb_img = np.zeros((h, w, 3))
    rgb_img[:, :, 0] = img
    rgb_img[:, :, 1] = img
    rgb_img[:, :, 2] = img

    return rgb_img


def main():

    val_data_json = json.load(open('cocoqa_data_prepro_93.json', 'r'))

    unique_img_val = val_data_json['unique_img_val']

    val_data_h5 = h5py.File('cocoqa_data_prepro_93.h5', 'r')

    img_pos_val = val_data_h5['img_pos_val'][:]

    img_data = imgs2batch(unique_img_val, img_pos_val)

    print(len(unique_img_val))
    print(len(img_data))

    file = open('img_data_' + str(len(unique_img_val)) + '.pkl', 'wb')
    pickle.dump(img_data, file)

    file.close()

if __name__ == '__main__':
    main()
