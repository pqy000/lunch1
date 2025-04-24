# tasks.py

import random
from torch.utils.data import Dataset

def get_cifar10_tasks():
    # 10类 -> 5个task, 每task 2类
    return [
        [0,1], [2,3], [4,5], [6,7], [8,9]
    ]

def get_cifar100_tasks():
    # 100类 -> 20个task, 每task 5类
    splits = []
    num_tasks = 20
    classes_per_task = 5
    for t in range(num_tasks):
        start = t * classes_per_task
        cset = list(range(start, start+classes_per_task))
        splits.append(cset)
    return splits

class TaskSubset(Dataset):
    def __init__(self, base_dataset, class_ids):
        super().__init__()
        self.samples = []
        self.class_ids_map = {c:i for i,c in enumerate(class_ids)}
        self.transform = base_dataset.transform

        for i in range(len(base_dataset)):
            img, label = base_dataset[i]
            if label in class_ids:
                self.samples.append((img,label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, old_label = self.samples[idx]
        new_label = self.class_ids_map[old_label]
        return img, new_label