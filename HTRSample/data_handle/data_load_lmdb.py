import pickle
import random
from collections import namedtuple
from typing import Tuple

import lmdb
import numpy as np
from path import Path

Sample = namedtuple('Sample', 'gt_text, file_name')
Batch = namedtuple('Batch', 'imgs, gt_texts, batch_size, file_names')

class DataLoadLmdb:
    def __init__(self, dataset_path: Path, batch_size: int = 500, data_split: float = 0.70) -> None:

        self.env = lmdb.open(str(dataset_path / 'lmdb'), readonly=True)
        self.data_augmentation = False
        self.curr_idx = 0
        self.batch_size = batch_size
        self.samples = []

        f = open(dataset_path / 'words.txt')
        chars = set()
        bad_samples_reference = ['a01-117-05-02', 'r06-022-03-05']  # broken images in IAM dataset
        for line in f:
            if not line or line[0] == '#': # comment line
                continue
            line_split = line.strip().split(' ')
            assert len(line_split) >= 9

            file_name_split = line_split[0].split('-')
            file_name_subdir1 = file_name_split[0]
            file_name_subdir2 = f'{file_name_split[0]}-{file_name_split[1]}'
            file_base_name = line_split[0] + '.png'
            file_name = dataset_path / 'wordImages' / file_name_subdir1 / file_name_subdir2 / file_base_name

            if line_split[0] in bad_samples_reference:
                print('Ignoring known broken image:', file_name)
                continue

            gt_text = ' '.join(line_split[8:])  # word are at 9 columns
            chars = chars.union(set(list(gt_text)))

            # put sample into list
            self.samples.append(Sample(gt_text, file_name))

        # split training and validation set
        split_idx = int(data_split * len(self.samples))
        self.train_samples = self.samples[:split_idx]
        self.validation_samples = self.samples[split_idx:]

        self.train_words = [x.gt_text for x in self.train_samples]
        self.validation_words = [x.gt_text for x in self.validation_samples]
        self.train_set()
        self.char_list = sorted(list(chars))

    def train_set(self) -> None:
        self.data_augmentation = True
        self.curr_idx = 0
        random.shuffle(self.train_samples)
        self.samples = self.train_samples
        self.curr_set = 'train'

    def validation_set(self) -> None:
        self.data_augmentation = False
        self.curr_idx = 0
        self.samples = self.validation_samples
        self.curr_set = 'val'

    def get_iterator_info(self) -> Tuple[int, int]:
        if self.curr_set == 'train':
            num_batches = int(np.floor(len(self.samples) / self.batch_size))
        else:
            num_batches = int(np.ceil(len(self.samples) / self.batch_size))
        curr_batch = self.curr_idx // self.batch_size + 1
        return curr_batch, num_batches

    def has_next(self) -> bool:
        if self.curr_set == 'train':
            return self.curr_idx + self.batch_size <= len(self.samples)
        else:
            return self.curr_idx < len(self.samples)

    def _get_img(self, i: int) -> np.ndarray:
        with self.env.begin() as txn:
            basename = Path(self.samples[i].file_name).basename()
            data = txn.get(basename.encode("ascii"))
            img = pickle.loads(data)
        return img

    def get_next(self) -> Batch:
        batch_range = range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))

        imgs = [self._get_img(i) for i in batch_range]
        gt_texts = [self.samples[i].gt_text for i in batch_range]

        file_names = [self.samples[i].file_name for i in batch_range]

        self.curr_idx += self.batch_size
        return Batch(imgs, gt_texts, len(imgs), file_names)
