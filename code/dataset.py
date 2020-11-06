from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class BlightDataset(Dataset):
    def __init__(self, root, set_type, transform=None, cropping=None):
        if set_type in ['train', 'val', 'test']:
            self.dataset_dir = os.path.join(root, set_type)
        else:
            raise Exception('Not valid set_type')
        self.datapoints = []
        file_list=os.listdir(os.path.join(self.dataset_dir, 'class0'))
        self.datapoints+=[(os.path.join('class0', file), 0) for file in file_list]
        file_list=os.listdir(os.path.join(self.dataset_dir, 'class1'))
        self.datapoints += [(os.path.join('class1', file), 1) for file in file_list]
        self.transform=transform
        self.cropping=cropping

    def __getitem__(self, index):
        img_path, label = self.datapoints[index]
        img_as_img = Image.open(os.path.join(self.dataset_dir, img_path))

        if self.cropping:
            img_as_img = img_as_img.resize((256,256))
            img_as_img = img_as_img.crop(self.cropping)
        if self.transform:
            img_as_img=self.transform(img_as_img)
        return (img_as_img, label)

    def __len__(self):
        return len(self.datapoints)