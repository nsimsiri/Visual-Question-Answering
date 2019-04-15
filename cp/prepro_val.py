import json
import numpy as np
from nltk.tokenize import word_tokenize
import h5py
import sys

N_DATA_GENEREATE = 10

def prepro_question(imgs):
    # preprocess all the question
    # print('example processed tokens:')
    for i, data in enumerate(imgs):
        # .encode("utf-8")
        s = data['question']
        txt = word_tokenize(str(s).lower())
        data['processed_tokens'] = txt
        # if i < 10:
        #     # print(data['question'])
        #     # print(txt)
        # if i % 1000 == 0:
        #     sys.stdout.write("processing %d/%d (%.2f%% done)   \r" % (i, len(imgs), i * 100.0 / len(imgs)))
        #     sys.stdout.flush()
    return imgs


def build_vocab_question(imgs_val, include_map=False):
    # build vocabulary for question and answers.
    count_thr = 0

    # count up the number of words
    counts = {}
    for img in imgs_val:
        for w in img['processed_tokens']:
            counts[w] = counts.get(w, 0) + 1

    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    # print('top words and their counts:')
    # print('\n'.join(map(str, cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab),))
    print('number of <unk>: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))

    # lets now produce the final annotation
    # additional special UNK token we will use below to map infrequent words to
    print('inserting the special UNK token')
    
    vocab.append('<unk>')
    vocab.append('<start>')
    vocab.append('<end>')
    vocab.append('<pad>')

    for img in imgs_val:
        txt = img['processed_tokens']
        question = [w if counts.get(w, 0) > count_thr else '<unk>' for w in txt]
        question = ['<start>'] + question + ['<end>']
        print question
        img['final_question'] = question

    if include_map:
        itow = {i:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
        wtoi = {w:i for i,w in enumerate(vocab)} # inverse table
        return imgs_val, vocab, itow, wtoi 
    return imgs_val, vocab


def get_top_answers(imgs_val):
    counts = {}

    for img in imgs_val:
        ans = img['ans']
        counts[ans] = counts.get(ans, 0) + 1

    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    # print('top answer and their counts:')
    # print('\n'.join(map(str, cw[:20])))
    vocab = []
    for i in range(len(cw)):
        vocab.append(cw[i][1])

    return vocab[:len(cw)]


def encode_question(imgs_val, wtoi):
    max_length = 26

    N = len(imgs_val)
    label_arrays_val = np.zeros((N, max_length), dtype='uint32')
    label_length_val = np.zeros(N, dtype='uint32')
    question_id_val = np.zeros(N, dtype='uint32')
    question_counter_val = 0
    for i, img in enumerate(imgs_val):
        question_id_val[question_counter_val] = img['ques_id']
        label_length_val[question_counter_val] = min(max_length,
                                                     len(img['final_question']))  # record the length of this sequence
        question_counter_val += 1
        for k, w in enumerate(img['final_question']):
            if k < max_length:
                label_arrays_val[i, k] = wtoi[w]


    return label_arrays_val, label_length_val, question_id_val

def encode_answer(imgs, atoi):
    N = len(imgs)
    ans_arrays = np.zeros(N, dtype='uint32')

    for i, img in enumerate(imgs):
        ans_arrays[i] = atoi.get(img['ans'], -1) # -1 means wrong answer.

    return ans_arrays


def get_unqiue_img(imgs):
    count_img = {}
    N = len(imgs)
    img_pos = np.zeros(N, dtype='uint32')
    for img in imgs:
        count_img[img['img_path']] = count_img.get(img['img_path'], 0) + 1

    unique_img = [w for w, n in count_img.items()]
    imgtoi = {w: i for i, w in enumerate(unique_img)}  # add one for torch, since torch start from 1.

    for i, img in enumerate(imgs):
        idx = imgtoi.get(img['img_path'])
        img_pos[i] = idx

    return unique_img, img_pos


val_data = json.load(open('vqa_raw_val.json', 'r'))

print('val_data.lengh', len(val_data))

val_data = val_data[:N_DATA_GENEREATE]

# print(val_data)
# 'ques_id': 458752000,
# 'img_path': 'train2014/COCO_train2014_000000458752.jpg',
# 'question': 'What is this photo taken looking through?',
# 'question_type': 'what is this',
# 'ans': 'net'

# {'ques_id': 262144000,
# 'img_path': 'test2015/COCO_test2015_000000262144.jpg',
# 'question': 'Is the ball flying towards the batter?'}

top_ans = get_top_answers(val_data)
atoi = {w: i for i, w in enumerate(top_ans)} # a 1-indexed vocab translation table
itoa = {i: w for i, w in enumerate(top_ans)} # inverse table

print 'atoi\n', atoi
print 'itoa\n', itoa
#tokenization and preprocessing val question
val_data = prepro_question(val_data)



# create the vocab for question
val_data, vocab = build_vocab_question(val_data)

itow = {i:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
wtoi = {w:i for i,w in enumerate(vocab)} # inverse table
print 'itow\n', atoi
print 'wtoi\n', itoa

sys.exit()


ques_val, ques_length_val, question_id_val,= encode_question(val_data, wtoi)

unique_img_val, img_pos_val = get_unqiue_img(val_data)

# get the answer encoding.
ans_val = encode_answer(val_data, atoi)



# create output h5 file for training set.
f = h5py.File('cocoqa_data_prepro.h5', "w")
f.create_dataset("ques_val", dtype='uint32', data=ques_val)

f.create_dataset("ans_val", dtype='uint32', data=ans_val)

f.create_dataset("question_id_val", dtype='uint32', data=question_id_val)

f.create_dataset("img_pos_val", dtype='uint32', data=img_pos_val)

f.create_dataset("ques_length_val", dtype='uint32', data=ques_length_val)

f.close()
print('wrote ', 'cocoqa_data_prepro.h5')

# create output json file
out = {}
out['ix_to_word'] = itow # encode the (1-indexed) vocab
out['ix_to_ans'] = itoa
out['unique_img_val'] = unique_img_val
json.dump(out, open('cocoqa_data_prepro.json', 'w'))
print('wrote ', 'cocoqa_data_prepro.json')