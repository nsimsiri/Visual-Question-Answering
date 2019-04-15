import json
from nltk.tokenize import word_tokenize


def generate_json_data():

    '''
    Put the VQA data into single json file, where [[Question_id, Image_id, Question, multipleChoice_answer, Answer] ... ]
    '''

    num_data = 100
    num_val = 0

    val = []

    imdir = '%s/COCO_%s_%012d.jpg'
    val_anno = json.load(open('data/v2_mscoco_val2014_annotations.json', 'r'))
    val_ques = json.load(open('data/v2_OpenEnded_mscoco_val2014_questions.json', 'r'))

    print(len(val_anno['annotations']))

    subtype = 'val2014'
    # for i in range(len(val_anno['annotations'])):
    for i in range(num_data):
        ans = val_anno['annotations'][i]['multiple_choice_answer']
        if len(word_tokenize(str(ans).lower())) == 1:
            question_id = val_anno['annotations'][i]['question_id']
            question_type = val_anno['annotations'][i]['question_type']
            image_path = imdir % (subtype, subtype, val_anno['annotations'][i]['image_id'])

            question = val_ques['questions'][i]['question']
            val.append(
                {'ques_id': question_id, 'img_path': image_path, 'question': question, 'question_type': question_type, 'ans': ans})

            num_val += 1

    print('Validation sample %d...' % (len(val)))
    json.dump(val, open('vqa_raw_val_' + str(num_val) + '.json', 'w'))


if __name__ == "__main__":
    generate_json_data()
