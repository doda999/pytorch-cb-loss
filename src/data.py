import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
import tensorflow as tf

HEIGHT = 32
WIDTH = 32
DEPTH = 3


def parse(x):
        image_feature_description = {
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64),
            }
        x = tf.io.parse_single_example(x, image_feature_description)
        image = tf.io.decode_raw(x["image"], tf.uint8)
        label = x['label']
        return (image, label)


class CifarDataset(Dataset):
    def __init__(self, cfg, mode="train"):
        assert mode in ["train", "eval"]
        self.mode = mode
        self.data_path = os.path.join(cfg.data_dir, mode+".tfrecords")
        self.num_labels = int(cfg.data_dir.split('/')[-1].split('-')[1])
        self.images = []
        self.labels = []
        self.label_counts = [0] * self.num_labels
        self.augmentation = transforms.Compose([
            transforms.RandomCrop((HEIGHT, WIDTH), 4),
            transforms.RandomHorizontalFlip(),
        ])
        self.normal_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])
        self.load_data()
        
    def load_data(self):
        raw_dataset = tf.data.TFRecordDataset(self.data_path)
        dataset = raw_dataset.map(parse)
        for data in dataset:
            image = data[0].numpy().reshape(DEPTH, HEIGHT, WIDTH).transpose(1,2,0) # C × H × W -> H × W × C
            label = data[1].numpy()
            self.images.append(Image.fromarray(image))
            self.labels.append(label)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if self.mode == "train":
            image = self.augmentation(image)
        return self.normal_transform(image), self.labels[index]
    
    def get_num_per_cls(self):
        if sum(self.label_counts):
            return self.label_counts
        for cls in range(self.num_labels):
            self.label_counts[cls] = (torch.tensor(self.labels) == cls).sum().item()
        return self.label_counts


