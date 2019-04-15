import json


def generate_json_data():

    '''
    Put the VQA data into single json file, where [[Question_id, Image_id, Question, multipleChoice_answer, Answer] ... ]
    '''

    train = []
    val = []
    test = []

    imdir = '%s/COCO_%s_%012d.jpg'

    train_anno = json.load(open('data/v2_mscoco_train2014_annotations.json', 'r'))
    val_anno = json.load(open('data/v2_mscoco_val2014_annotations.json', 'r'))

    train_ques = json.load(open('data/v2_OpenEnded_mscoco_train2014_questions.json', 'r'))
    val_ques = json.load(open('data/v2_OpenEnded_mscoco_val2014_questions.json', 'r'))
    test_ques = json.load(open('data/v2_OpenEnded_mscoco_test2015_questions.json', 'r'))

    subtype = 'train2014'
    for i in range(len(train_anno['annotations'])):
        ans = train_anno['annotations'][i]['multiple_choice_answer']
        question_id = train_anno['annotations'][i]['question_id']
        question_type = train_anno['annotations'][i]['question_type']
        image_path = imdir % (subtype, subtype, train_anno['annotations'][i]['image_id'])

        question = train_ques['questions'][i]['question']
        train.append(
            {'ques_id': question_id, 'img_path': image_path, 'question': question, 'question_type': question_type, 'ans': ans})

    subtype = 'val2014'
    for i in range(len(val_anno['annotations'])):
        ans = val_anno['annotations'][i]['multiple_choice_answer']
        question_id = val_anno['annotations'][i]['question_id']
        question_type = val_anno['annotations'][i]['question_type']
        image_path = imdir % (subtype, subtype, val_anno['annotations'][i]['image_id'])

        question = val_ques['questions'][i]['question']
        val.append(
            {'ques_id': question_id, 'img_path': image_path, 'question': question, 'question_type': question_type, 'ans': ans})

    subtype = 'test2015'
    for i in range(len(test_ques['questions'])):
        question_id = test_ques['questions'][i]['question_id']
        image_path = imdir % (subtype, subtype, test_ques['questions'][i]['image_id'])

        question = test_ques['questions'][i]['question']
        test.append(
            {'ques_id': question_id, 'img_path': image_path, 'question': question})

    print('Training sample %d, Validation sample %d, Testing sample %d...' % (len(train), len(val), len(test)))

    json.dump(train, open('vqa_raw_train.json', 'w'))
    json.dump(val, open('vqa_raw_val.json', 'w'))
    json.dump(test, open('vqa_raw_test.json', 'w'))


if __name__ == "__main__":
    generate_json_data()
