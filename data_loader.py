import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.utils.data as Data

import json
import os, sys, re

from PIL import Image
import json, time
import numpy as np
import pandas as pd
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
from random import shuffle
import random
from collections import defaultdict
from nltk.tokenize import word_tokenize
from utils import imread
import h5py
RANDOM_SEED = 9001
ann_path = 'data/v2_mscoco_val2014_annotations.json'
q_path   = 'data/v2_OpenEnded_mscoco_val2014_questions.json'
i_path   = 'data/val2014'
i_prefix = 'COCO_val2014_'
DEBUG = False
PP = lambda parsed: print(json.dumps(parsed, indent=4, sort_keys=True))
def clean(words):
    # token = re.sub(r'\W+', '', word)
    tokens = words.lower()
    tokens = word_tokenize(tokens)
    return tokens

def clean_answer(answer):
    token = re.sub(r'\W+', '', answer)
    token = clean(token)
    if (len(token)>1): return None
    return token[0]

def collate_sort_by_q_wrap(dataset):
    def collate_sort_by_q(minibatch):
        max_seq_len = 0
        minibatch.sort(key=lambda minibatch_tuple: minibatch_tuple[-1], reverse=True)
        for row in minibatch:
            idx, v,q,a,l = row
            max_seq_len = max(max_seq_len, l)
        for row in minibatch:
            idx, v,q,a,l = row
            q += [dataset.qtoi['<pad>'] for _i in range(max_seq_len-len(q))]
        
        return Data.dataloader.default_collate(minibatch)
    return collate_sort_by_q

class VQADataSet():
    def __init__(self, ann_path=ann_path, ques_path=q_path, img_path=i_path, 
                 TEST_SPLIT=0.2, Q=5):
        t0 = time.time()
        self.answer_maps = []
        self.question_maps = {}
        self.splits = {'train':[], 'test':[]}
        self.Q = Q
        self.ann_path = ann_path
        self.quest_path = ques_path
        self.img_path = img_path

        self.special_tokens = ['<pad>','<start>', '<end>', '<unk>']
        self.itoa, self.atoi = [], {}
        self.itoq, self.qtoi = [], {}
        self.vocab = {'answer': [] ,'question': []}
        self.max_length = -1
        self.TEST_SPLIT = TEST_SPLIT
        
        self.qdf = None # Panda Frame of questions
        self.anns = None # List of annotations (with answers, quesiton_id, image_id)
 
        ### Load Dataset ###
        q_json = None
        with open(q_path, 'r') as q_f:
            q_json = json.load(q_f);
        self.qdf = pd.DataFrame(q_json['questions'])

        with open(ann_path, 'r') as ann_f:
            self.anns = json.load(ann_f)['annotations']

        ### Initialize Data ###
        if (self.Q == -1):
            self.Q = len(self.anns)
        self._init_qa_maps()
        self._build_vocab()
        self._encode_qa_and_set_img_path()
        self._randomize_equally_distributed_splits()
        del self.anns
        # if DEBUG:
        print('VQADataSet init time: {}'.format(time.time() - t0))
    
    @staticmethod
    def batchify_questions(q):
        return torch.stack(q).t()

    def build_data_loader(self, train=False, test=False, args=None):
        if (args is None):
            args = {'batch_size': 32}

        if test:
            args['shuffle'] = False
        elif train:
            args['shuffle'] = True
        batch_size = args['batch_size']
        shuffle    = args['shuffle']
        print('batch_size: {} shuffle: {}'.format(batch_size, shuffle))
        data_loader_split = VQADataLoader(self, train=train, test=test)
        data_generator = Data.DataLoader(dataset=data_loader_split, 
                                         batch_size=batch_size, 
                                         shuffle=shuffle,
                                         collate_fn=collate_sort_by_q_wrap(self))
        return data_generator

    # set qdf, question_maps
    def _init_qa_maps(self):
        cnt = 0 
        for ann_idx in tqdm(range(self.Q)):
            ann = self.anns[ann_idx];
            answer_set = set()
            answers = []
            question_id = ann['question_id']
            for ans in ann['answers']:
                ans_text = ans['answer']
                
                ans_tokens = clean(ans_text)
                if (len(ans_tokens) != 1): continue
                ans_text = ans_tokens[0] 
                if ans_text not in answer_set:
                    ans['question_id'] = question_id
                    answers.append(ans)
                answer_set.add(ans_text)

            if (len(answers) == 0): continue
            question = self.qdf.query('question_id == {}'.format(question_id))

            self.answer_maps += answers
            self.question_maps[question_id] = question.to_dict(orient='records')[0]

            if (cnt >= self.Q): break
            cnt+=1

    def _build_vocab(self):
        q_vocab = set()
        a_vocab = set()
        if DEBUG: print('build answer vocab')
        for ann in tqdm(self.answer_maps):
            answer = ann['answer']
            # answer_tokens = clean(answer)
            # ann['tokens'] = answer_tokens
            ann['tokens'] = [answer]
            a_vocab.add(answer)
        if DEBUG: print('build question vocab)')
        for question_id, question_json in tqdm(self.question_maps.items()):
            question = question_json['question']
            question_tokens = clean(question)
            question_json['tokens'] = ["<start>"] + question_tokens + ["<end>"]
            self.max_length = max(len(question_json['tokens']), self.max_length)
            q_vocab = q_vocab.union(set(question_tokens))
        
        q_vocab_list = self.special_tokens + list(q_vocab)
        a_vocab_list = list(a_vocab)

        self.vocab['answer'] = a_vocab_list
        self.vocab['question'] = q_vocab_list
        self.itoq = self.vocab['question']
        self.itoa = self.vocab['answer']
        self.qtoi = {q: i for i,q in enumerate(q_vocab_list)}
        self.atoi = {a: i for i,a in enumerate(a_vocab_list)}
        
    def _encode_qa_and_set_img_path(self):
        if DEBUG: print('encode answers')
        for ann in tqdm(self.answer_maps):
            a_tokens = ann['tokens']
            ann['encoding'] = [self.atoi[w]for w in a_tokens]
        if DEBUG: print('encode questions')
        for question_id, question_json in tqdm(self.question_maps.items()):
            image_id = question_json['image_id']
            q_tokens = question_json['tokens']
            question_json['encoding'] = [self.qtoi[w] for w in q_tokens]
            question_json['image_path'] = self._img_id_to_path(str(image_id))
        
    def _img_id_to_path(self, img_id):
        eg = '000000000192'
        total = len(eg)
        full_img_id = '0'*(total-len(img_id)) + img_id
        img_f = i_path + "/" + i_prefix + full_img_id + ".jpg"
        img_f = img_f.strip()
        return img_f

    def _randomize_equally_distributed_splits(self):
        cntr = defaultdict(int)
        dist = defaultdict(list)
        for i, ann in enumerate(self.answer_maps):
            ans = ann['answer']
            cntr[ans]+=1
            dist[ans].append(i)

        splits = {'train': [], 'test': []}
        z_cnt = 0
        for ans, idxes in tqdm(dist.items()):
            random.Random(RANDOM_SEED).shuffle(idxes)
            c = int(len(idxes)*self.TEST_SPLIT)
            splits['train'] += idxes[c:]
            splits['test'] += idxes[:c]
        sorted(splits['train'])
        sorted(splits['test'])
        self.splits = splits

    def __len__(self):
        return len(self.answer_maps)

    def get(self, idx, split_type):
        v,q,a = None, None, None
        try:
            split_keys = self.splits[split_type]
            ans_key = split_keys[idx]
            answer_json = self.answer_maps[ans_key]
            question_key = answer_json['question_id']
            question_json = self.question_maps[question_key]
        except:
            print("ERR")
        return question_json, answer_json
    
    def size(self):
        return (len(self.question_maps), len(self.answer_maps))

    def get_max_sequence_len(self):
        return self.max_length

    def decode_question(self, encoding):
        for x in encoding:
            if x < 0 or x >= len(self.itoq):
                raise Exception("DECODE_ERR: cannot find word-idx: {}".format(x))
        sen_vec = [self.itoq[x] for x in encoding]
        sen = " ".join(sen_vec)
        return sen
    
    def decode_answer(self, encoding):
        if encoding < 0 or encoding >= len(self.itoa):
            raise Exception("DECODE_ERR: cannot find word-idx: {}".format(encoding))
        return self.itoa[encoding] 

class VQADataLoader(data.Dataset):
    def __init__(self, dataset, train=False, test=False):
        assert(train+test==1)
        split_type = None
        if train: split_type = 'train'
        elif test: split_type = 'test'
        self.split_keys = dataset.splits[split_type]
        self.dataset = dataset
        self.split_type=split_type

    def __len__(self):
        return len(self.split_keys)
    
    '''
    Returns:
        v: torch.Size([BATCH_SIZE, 3, 224, 224])
        q: [tensor(a_0, a_1,...), tensor(a_0..)]
        a: tensor([ans_1, ans_2,...])
        q_len: tensor([len_1, len_2,...])
,
    '''
    def __getitem__(self, idx):
        v,q,a = -1, -1, -1
        try:
            question_json, answer_json = self.dataset.get(idx, self.split_type)
            img_path = question_json['image_path']
            v     = imread(img_path)
            q     = question_json['encoding']
            a     = answer_json['encoding'][0]
            q_len = len(q)
        except Exception as e:
            print("DATALOAD-ERR: " + str(e))
        return idx, v, q, a, q_len
    

       
if __name__ == '__main__':
    print("hello")

