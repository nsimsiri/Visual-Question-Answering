import json
import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize
import h5py

data_type = 'uint8'

def prepro_question(imgs):
    for i, data in enumerate(imgs):
        s = data['question']
        txt = word_tokenize(str(s).lower())
        data['processed_tokens'] = txt
    return imgs


def build_vocab_question(imgs_val, include_map=False):
    # build vocabulary for question and answers.
    count_thr = 0

    # count up the number of words
    counts = {}
    for img in imgs_val:
        for w in img['processed_tokens']:
            counts[w] = counts.get(w, 0) + 1

    # cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    # print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    # print('number of words in vocab would be %d' % (len(vocab),))
    # print('number of <unk>: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))
    
    vocab.append('<unk>')
    vocab.append('<start>')
    vocab.append('<end>')
    vocab.append('<pad>')

    for img in imgs_val:
        txt = img['processed_tokens']
        question = [w if counts.get(w, 0) > count_thr else '<unk>' for w in txt]
        question = ['<start>'] + question + ['<end>']
        # print(question)
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

    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    # print('top answer and their counts:')
    # print('\n'.join(map(str, cw[:20])))
    vocab = []
    for i in range(len(cw)):
        vocab.append(cw[i][1])

    return vocab[:len(cw)]


def encode_question(imgs_val, wtoi):
    max_length = 26

    N = len(imgs_val)
    label_arrays_val = np.zeros((N, max_length), dtype=data_type)
    label_length_val = np.zeros(N, dtype=data_type)
    question_id_val = np.zeros(N, dtype=data_type)
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


def encode_answer(imgs_val, atoi):
    N = len(imgs_val)
    ans_arrays = np.zeros(N, dtype='uint32')

    for i, img in enumerate(imgs_val):
        ans_arrays[i] = atoi.get(img['ans'], -1)  # -1 means wrong answer.

    return ans_arrays


def get_unqiue_img(imgs_val):
    count_img = {}
    N = len(imgs_val)
    img_pos = np.zeros(N, dtype=data_type)
    for img in imgs_val:
        count_img[img['img_path']] = count_img.get(img['img_path'], 0) + 1

    unique_img = [w for w, n in count_img.items()]
    imgtoi = {w: i for i, w in enumerate(unique_img)}  # add one for torch, since torch start from 1.

    for i, img in enumerate(imgs_val):
        idx = imgtoi.get(img['img_path'])
        img_pos[i] = idx

    return unique_img, img_pos

# =========================================================================

if __name__ == '__main__':
    N_DATA_GENEREATE = 93

    val_data = json.load(open('vqa_raw_val_93.json', 'r'))

    print('val_data:', len(val_data))

    val_data = val_data[:N_DATA_GENEREATE]

    print(val_data)
    print(' ')

    # 'ques_id': 262148000,
    # 'img_path': 'val2014/COCO_val2014_000000262148.jpg',
    # 'question': 'Where is he looking?',
    # 'question_type': 'none of the above',
    # 'ans': 'down'}

    top_ans = get_top_answers(val_data)
    atoi = {w: i+1 for i, w in enumerate(top_ans)}
    itoa = {i+1: w for i, w in enumerate(top_ans)}

    print(top_ans)
    print('atoi\n', atoi)
    print('itoa\n', itoa)

    val_data = prepro_question(val_data)

    val_data, vocab = build_vocab_question(val_data)
    itow = {i+1:w for i,w in enumerate(vocab)}
    wtoi = {w:i+1 for i,w in enumerate(vocab)}

    print('itow\n', itow)
    print('wtoi\n', wtoi)

    print(val_data)

    ques_val, ques_length_val, question_id_val = encode_question(val_data, wtoi)

    print(ques_val)

    unique_img_val, img_pos_val = get_unqiue_img(val_data)

    print(unique_img_val)
    print(img_pos_val)

    # get the answer encoding.
    ans_val = encode_answer(val_data, atoi)

    print(' ')

    # print(ans_val)

    # print(question_id_val)
    # print(img_pos_val)
    # print(ques_length_val)
    # print(unique_img_val)


    # =========================================================================


    h5py_name = 'cocoqa_data_prepro_' + str(len(ques_val)) + '.h5'
    json_name = 'cocoqa_data_prepro_' + str(len(ques_val)) + '.json'

    f = h5py.File(h5py_name, "w")

    f.create_dataset("ques_val", dtype=data_type, data=ques_val)

    f.create_dataset("ans_val", dtype=data_type, data=ans_val)

    f.create_dataset("question_id_val", dtype=data_type, data=question_id_val)

    f.create_dataset("img_pos_val", dtype=data_type, data=img_pos_val)

    f.create_dataset("ques_length_val", dtype=data_type, data=ques_length_val)

    f.close()

    print('wrote: ' + str(len(ques_val)), h5py_name)

    out = {}

    out['ix_to_word'] = itow
    out['ix_to_ans'] = itoa
    out['unique_img_val'] = unique_img_val

    json.dump(out, open(json_name, 'w'))

    print('wrote ' + str(len(ques_val)), json_name)
