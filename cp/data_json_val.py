import json


def generate_json_data():

    '''
    Put the VQA data into single json file, where [[Question_id, Image_id, Question, multipleChoice_answer, Answer] ... ]
    '''

    train = []
    val = []
    test = []

    imdir = '%s/COCO_%s_%012d.jpg'

    val_anno = json.load(open('data/v2_mscoco_val2014_annotations.json', 'r'))
    val_ques = json.load(open('data/v2_OpenEnded_mscoco_val2014_questions.json', 'r'))


    subtype = 'val2014'
    for i in range(len(val_anno['annotations'])):
        ans = val_anno['annotations'][i]['multiple_choice_answer']
        question_id = val_anno['annotations'][i]['question_id']
        question_type = val_anno['annotations'][i]['question_type']
        image_path = imdir % (subtype, subtype, val_anno['annotations'][i]['image_id'])
            
        question = val_ques['questions'][i]['question']
        val.append(
            {'ques_id': question_id, 'img_path': image_path, 'question': question, 'question_type': question_type, 'ans': ans})


    print('Validation sample %d...' % (len(val)))
    json.dump(val, open('vqa_raw_val.json', 'w'))


if __name__ == "__main__":
    generate_json_data()
