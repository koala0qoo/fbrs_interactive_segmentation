import json
import random
from collections import defaultdict
from pathlib import Path
import pickle
import torch
from torchvision import transforms


import cv2
import numpy as np

class ISDataset(torch.utils.data.dataset.Dataset):
    def __init__(self,
                 points_from_one_object=True,
                 augmentator=None,
                 num_masks=1,
                 input_transform=None,
                 image_rescale=None,
                 min_object_area=0,
                 min_ignore_object_area=10,
                 keep_background_prob=0.0,
                 with_image_info=False,
                 samples_scores_path=None,
                 samples_scores_gamma=1.0,
                 zoom_in=None,
                 epoch_len=-1):
        super(ISDataset, self).__init__()
        self.epoch_len = epoch_len
        self.num_masks = num_masks
        self.points_from_one_object = points_from_one_object
        self.input_transform = input_transform
        self.augmentator = augmentator
        self.image_rescale = image_rescale
        self.min_object_area = min_object_area
        self.min_ignore_object_area = min_ignore_object_area
        self.keep_background_prob = keep_background_prob
        self.with_image_info = with_image_info
        self.samples_precomputed_scores = self._load_samples_scores(samples_scores_path, samples_scores_gamma)
        self.zoom_in = zoom_in
        if isinstance(self.image_rescale, (float, int)):
            scale = self.image_rescale
            self.image_rescale = lambda shape: scale

        if input_transform is None:
            input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
            ])
        self.input_transform = input_transform

        self.dataset_samples = None
        self._precise_masks = ['instances_mask']
        self._from_dataset_mapping = None
        self._to_dataset_mapping = None

    def __getitem__(self, index):
        if self.samples_precomputed_scores is not None:
            index = np.random.choice(self.samples_precomputed_scores['indices'],
                                     p=self.samples_precomputed_scores['probs'])
        else:
            if self.epoch_len > 0:
                index = random.randrange(0, len(self.dataset_samples))

        sample = self.get_sample(index)

        if use_zoom_in:
            sample = self.zoom_in(sample)
        else:
            sample = self.exclude_small_objects(sample)

        sample['objects_ids'] = [obj_id for obj_id, obj_info in sample['instances_info'].items()
                                 if not obj_info['ignore']]

        return 0

    def check_zoom_in(self, sample):
        use_zoom_in = self.zoom_in is not None and random.random() < self.zoom_in.p
        if use_zoom_in:
            sample = self.exclude_small_objects(sample)
            num_objects = len([x for x in sample['instances_info'].values() if not x['ignore']])
            if num_objects == 0:
                use_zoom_in = False

        return sample, use_zoom_in

    def augment_sample(self, sample, use_zoom_in=False):
        augmentator = self.augmentator if not use_zoom_in else self.zoom_in.augmentator
        if augmentator is None:
            return sample

        masks_to_augment = [mask_name for mask_name in self._precise_masks if mask_name in sample]
        masks = [sample[mask_name] for mask_name in masks_to_augment]

        valid_augmentation = False
        while not valid_augmentation:
            aug_output = augmentator(image=sample['image'], masks=masks)
            valid_augmentation = self.check_augmented_sample(sample, aug_output, masks_to_augment)

        sample['image'] = aug_output['image']
        for mask_name, mask in zip(masks_to_augment, aug_output['masks']):
            sample[mask_name] = mask

        sample_ids = set(get_unique_labels(sample['instances_mask'], exclude_zero=True))
        instances_info = sample['instances_info']
        instances_info = {sample_id: sample_info for sample_id, sample_info in instances_info.items()
                          if sample_id in sample_ids}
        sample['instances_info'] = instances_info

        return sample

    def check_augmented_sample(self, sample, aug_output, masks_to_augment):
        if self.keep_background_prob < 0.0 or random.random() < self.keep_background_prob:
            return True

        aug_instances_mask = aug_output['masks'][masks_to_augment.index('instances_mask')]
        aug_sample_ids = set(get_unique_labels(aug_instances_mask, exclude_zero=True))
        num_objects_after_aug = len([obj_id for obj_id in aug_sample_ids
                                     if not sample['instances_info'][obj_id]['ignore']])

        return num_objects_after_aug > 0

    def exclude_small_objects(self, sample):
        if self.min_object_area <= 0:
            return sample

        for obj_id, obj_info in sample['instances_info'].items():
            if not obj_info['ignore']:
                obj_area = (sample['instances_mask'] == obj_id).sum()
                if obj_area < self.min_object_area:
                    obj_info['ignore'] = True

        return sample

    @staticmethod
    def _load_samples_scores(samples_scores_path, samples_scores_gamma):
        if samples_scores_path is None:
            return None

        with open(samples_scores_path, 'rb') as f:
            images_scores = pickle.load(f)

        probs = np.array([(1.0 - x[2]) ** samples_scores_gamma for x in images_scores])
        probs /= probs.sum()
        samples_scores = {
            'indices': [x[0] for x in images_scores],
            'probs': probs
        }
        print(f'Loaded {len(probs)} weights with gamma={samples_scores_gamma}')
        return samples_scores

    def get_sample(self, index):
        raise NotImplementedError

    def __len__(self):
        if self.epoch_len > 0:
            return self.epoch_len
        else:
            return len(self.dataset_samples)


def get_unique_labels(x, exclude_zero=False):
    obj_sizes = np.bincount(x.flatten())
    labels = np.nonzero(obj_sizes)[0].tolist()

    if exclude_zero:
        labels = [x for x in labels if x != 0]
    return labels


class XrayDataset(ISDataset):

    CLASSES = [' ', 'metal_lighter', 'lighter', 'knife', 'battery', 'scissor']

    def __init__(self, dataset_path, split='train',
                 max_overlap_ratio=0.5,
                 **kwargs):
        super(XrayDataset, self).__init__(**kwargs)
        dataset_path = Path(dataset_path)
        self._split_path = dataset_path / split
        self.dataset_path = dataset_path
        self.split = split
        self.max_overlap_ratio = max_overlap_ratio

        with open(dataset_path / f'{self.split}_annotations.json', 'r') as f:
            json_annotation = json.loads(f.read())

        self.annotations = defaultdict(list)
        for x in json_annotation['annotations']:
            self.annotations[x['image_id']].append(x)

        self.dataset_samples = [x for x in json_annotation['images']
                                if len(self.annotations[x['id']]) > 0]

    def get_score(self, index):
        image_info = self.dataset_samples[index]
        image_id, image_filename = image_info['id'], image_info['file_name']
        num_instance = len(self.annotations[image_id])
        score = 1 - num_instance/30
        if score > 0.9:
            score = 0.9
        if score <= 0.1:
            score = 0.1
        print(image_filename, num_instance, score)

        return index, image_filename, score


trainset = XrayDataset(
        "/home/hui.yuan/data/tianchi/round2",
        split='train',
        points_from_one_object=False,
        samples_scores_path=None,
        samples_scores_gamma=1
    )
scores = []
for i in range(len(trainset)):
    score = trainset.get_score(i)
    scores.append(score)
    with open('xray_samples_weights.pkl', 'wb') as f:
        pickle.dump(scores, f)