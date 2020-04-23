import os
import numpy as np
import scipy.io
from torch.utils.data import Dataset
import torch


class Svhn(Dataset):
    def __init__(self,root_dir,transforms=None,split='train'):
        image_file = 'train_32x32.mat' if split == 'train' else 'test_32x32.mat'
        image_dir = os.path.join(root_dir, image_file)
        svhn = scipy.io.loadmat(image_dir)
        self.images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 255.
        # self.images = np.transpose(svhn['X'], [1, 0, 3, 2]) / 255.
        # self.images = svhn['X']/255.
        labels = svhn['y'].reshape(-1)
        labels[np.where(labels == 10)] = 0
        labels = labels.astype('int32')
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):

        image = self.images[index]
        label = self.labels[index]
        # label = torch.from_numpy(label)
        if self.transforms:
            image = self.transforms(image)
        return image, label.item()

    def __len__(self):
        return len(self.images)






