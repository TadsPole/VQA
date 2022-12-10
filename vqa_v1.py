import copy
import json
import os
from .DataLoaderX import DataLoaderX
import torch
import numpy as np
import random
from dataclasses import dataclass
from typing import Tuple
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from collections import defaultdict
from transformers import FlavaProcessor, BertTokenizerFast
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 2300000000

@dataclass
class VQA_v1Config:
    max_length: int = 64
    data_path: str = ''
    img_path: str = ''
    batch_size: int = 16
    num_workers: int = 20
    shuffle_seed: int = 10086


class VQA_v1Dataset(Dataset):
    def __init__(self, dataType, dataSubtype):
        super().__init__()
        self.dataType = dataType
        self.dataSubtype = dataSubtype
        self.imageDir = 'Images/%s/%s/' % (dataType, dataSubtype)
        quesFile = 'Questions/%s/%s_%s_questions.json' % (dataType, dataType, dataSubtype)
        with open(quesFile) as qp:
            self.questions = json.load(qp)
        if dataType != 'test':
            annFile = 'Annotations/%s/%s_%s_annotations.json' % (dataType, dataType, dataSubtype)
            with open(annFile) as ap:
                self.annotations = json.load(ap)

    def loadData(self) -> list:
        data = []
        question_id_list = [a['question_id'] for a in self.annotations['annotations']]
        i = 0
        for q in self.questions['questions']:
            print(i)
            i += 1
            image_id = q['image_id']
            question = q['question']
            question_id = q['question_id']
            multiple_choices = q['multiple_choices']
            if self.dataType != 'test':
                index = question_id_list.index(question_id)
                # find answer correlated to question id
                answer = self.annotations['annotations'][index]['multiple_choice_answer']
                # multiple choices contain correct answer, remove it
                multiple_choices=multiple_choices.remove(answer)
            else:
                answer = ''  # test questions have no given answers
            # match image id with filename
            image_filename = 'COCO_' + self.dataSubtype + '_' + str(image_id).zfill(12) + '.jpg'
            data.append({
                'image': os.path.join(self.imageDir, image_filename),
                'question': question,
                'multiple_choices': multiple_choices,
                'answer': answer
            })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

class VQA_v1Collate:
    def __init__(self, multi_processor: FlavaProcessor,
                 config: VQA_v1Config) -> None:
        self.multi_processor = multi_processor
        self.config = config

    def __call__(self, batch_samples):
        batch_question, batch_img, batch_answer,batch_choices = [], [], [], []
        for sample in batch_samples:
            batch_question.append(sample['question'])
            batch_img.append(Image.open(sample['image']).convert('RGB'))
            batch_answer.append(sample['answer'])
            batch_choices.append(sample['multiple_choices'])

        multi_input = self.multi_processor(text=batch_question,
                                           images=batch_img,
                                           return_tensors="pt",
                                           padding="max_length",
                                           max_length=160,
                                           truncation=True)
        labels = []
        for i in range(len(batch_question)):
            temp = [1]
            for _ in range(len(batch_choices[i])):
                temp.append(0)
            labels.append(temp)
        labels = torch.tensor(labels, dtype=int)
        return multi_input, labels


def getVQA_v1DataLoader(config: VQA_v1Config=VQA_v1Config()) \
        -> Tuple[DataLoaderX, DataLoaderX, DataLoaderX]:
    train_dataset = VQA_v1Dataset('mscoco','train')
    val_dataset = VQA_v1Dataset('mscoco','val')
    test_dataset = VQA_v1Dataset('mscoco','test')
    multi_processor = FlavaProcessor.from_pretrained('facebook/flava-full')
    collateFn = VQA_v1Collate(multi_processor, config)
    train_dataloader = DataLoaderX(train_dataset,
                                   batch_size=config.batch_size,
                                   collate_fn=collateFn,
                                   shuffle=True,
                                   num_workers=config.num_workers,
                                   pin_memory=True)
    val_dataloader = DataLoaderX(val_dataset,
                                 batch_size=config.batch_size,
                                 collate_fn=collateFn,
                                 shuffle=True,
                                 num_workers=config.num_workers,
                                 pin_memory=True)
    test_dataloader = DataLoaderX(test_dataset,
                                  batch_size=1,
                                  collate_fn=collateFn,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True)

    torch.manual_seed(config.shuffle_seed)
    torch.cuda.manual_seed_all(config.shuffle_seed)
    np.random.seed(config.shuffle_seed)
    random.seed(config.shuffle_seed)
    torch.backends.cudnn.deterministic = True
    return train_dataloader, val_dataloader, test_dataloader
