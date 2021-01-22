import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

dataset_path = '/home/hui.yuan/data/tianchi/round2/'

with open(f'{dataset_path}train_restriction.json', 'r') as f:
    json_annotation = json.loads(f.read())

train_annotations = dict()
train_annotations['images'] = []
train_annotations['type'] = 'instances'
train_annotations['annotations'] = json_annotation['annotations']
train_annotations['categories'] = json_annotation['categories']
test_annotations = dict()
test_annotations['images'] = []
test_annotations['type'] = 'instances'
test_annotations['annotations'] = json_annotation['annotations']
test_annotations['categories'] = json_annotation['categories']
count_train = 0
count_test = 0
for x in json_annotation['images']:
    if random.random() < 0.85:
        train_annotations['images'].append(x)
        count_train += 1
    else:
        test_annotations['images'].append(x)
        count_test += 1

with open(f'{dataset_path}train_annotations.json', 'w') as f:
    json.dump(train_annotations, f)
with open(f'{dataset_path}test_annotations.json', 'w') as f:
    json.dump(test_annotations, f)
print(count_train, count_test)
