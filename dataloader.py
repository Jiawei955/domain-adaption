import numpy as np

import torch
from torch.utils.data import Dataset



class SourceSet_Influence(Dataset):
    def __init__(self, domain_adaptation_task, repetition, sample_per_class):
        x_source_path = './row_data/' + domain_adaptation_task + '_X_train_source_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class) + '.npy'
        y_source_path = './row_data/' + domain_adaptation_task + '_y_train_source_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class) + '.npy'
        self.task = domain_adaptation_task
        self.x_source = np.load(x_source_path)
        self.y_source = np.load(y_source_path)

    def __getitem__(self, idx):
        x, y = self.x_source[idx], self.y_source[idx]
        if self.task == "MNIST_to_USPS" or self.task == "USPS_to_MNIST":
            x = torch.from_numpy(x).unsqueeze(0)
        else:
            x = torch.from_numpy(x)
        return x, y

    def __len__(self):
        return len(self.x_source)


class TargetSet_Influence(Dataset):
    def __init__(self, domain_adaptation_task, repetition, sample_per_class):
        x_target_path = './row_data/' + domain_adaptation_task + '_X_train_target_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class) + '.npy'
        y_target_path = './row_data/' + domain_adaptation_task + '_y_train_target_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class) + '.npy'
        self.task = domain_adaptation_task
        self.x_target = np.load(x_target_path)
        self.y_target = np.load(y_target_path)

    def __getitem__(self, idx):
        x, y = self.x_target[idx], self.y_target[idx]
        if self.task == "MNIST_to_USPS" or self.task == "USPS_to_MNIST":
            x = torch.from_numpy(x).unsqueeze(0)
        else:
            x = torch.from_numpy(x)
        return x, y

    def __len__(self):
        return len(self.x_target)

class TrainSet_Plain(Dataset):
    def __init__(self, domain_adaptation_task, repetition, sample_per_class, removed_indices=[], shuffle_target_labels=False):
        self.task = domain_adaptation_task
        x_source_path = './row_data/' + domain_adaptation_task + '_X_train_source_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class) + '.npy'
        y_source_path = './row_data/' + domain_adaptation_task + '_y_train_source_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class) + '.npy'

        self.x_source = np.load(x_source_path)
        self.y_source = np.load(y_source_path)

        x_target_path = './row_data/' + domain_adaptation_task + '_X_train_target_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class) + '.npy'
        y_target_path = './row_data/' + domain_adaptation_task + '_y_train_target_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class) + '.npy'

        self.x_target = np.load(x_target_path)
        self.y_target = np.load(y_target_path)

        if shuffle_target_labels:
            np.random.shuffle(self.y_target)

        if len(removed_indices) > 0:
            self.x_source = np.delete(self.x_source, removed_indices, 0)
            self.y_source = np.delete(self.y_source, removed_indices, 0)

        self.x_train = np.concatenate([self.x_source, self.x_target])
        self.y_train = np.concatenate([self.y_source, self.y_target])

        print(self.x_train.shape)


    def __getitem__(self, idx):
        x, y = self.x_train[idx], self.y_train[idx]
        if self.task == "MNIST_to_USPS" or self.task == "USPS_to_MNIST":
            x = torch.from_numpy(x).unsqueeze(0)
        else:
            x = torch.from_numpy(x)
        return x, y

    def __len__(self):
        return len(self.x_train)

# Initialization.Create_Pairs
class TrainSet(Dataset):
    def __init__(self, domain_adaptation_task, repetition, sample_per_class, weights=[], shuffle_target_labels=False):
        x_source_path = './row_data/' + domain_adaptation_task + '_X_train_source_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class) + '.npy'
        y_source_path = './row_data/' + domain_adaptation_task + '_y_train_source_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class) + '.npy'
        x_target_path = './row_data/' + domain_adaptation_task + '_X_train_target_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class) + '.npy'
        y_target_path = './row_data/' + domain_adaptation_task + '_y_train_target_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class) + '.npy'

        self.task = domain_adaptation_task
        self.x_source=np.load(x_source_path)
        self.y_source=np.load(y_source_path)
        self.x_target=np.load(x_target_path)
        self.y_target=np.load(y_target_path)

        if shuffle_target_labels:
            np.random.shuffle(self.y_target)

        num_classes = len(np.unique(self.y_source))
        prob_multi_for_P = (num_classes - 1) / 3


        if len(weights) == 0:
            weights = np.ones(self.x_source.shape[0])


        print("Source X : ", len(self.x_source), " Y : ", len(self.y_source))
        print("Target X : ", len(self.x_target), " Y : ", len(self.y_target))

        Training_P=[]
        Training_N=[]
        weights_P = []
        weights_N = []
        print('y_source :', self.y_source)
        for trs in range(len(self.y_source)):
            for trt in range(len(self.y_target)):
                if self.y_source[trs] == self.y_target[trt]:
                    Training_P.append([trs,trt, 1])
                    weights_P.append(weights[trs] * prob_multi_for_P)
                else:
                    Training_N.append([trs,trt, 0])
                    weights_N.append(weights[trs])
        print("Class P : ", len(Training_P), " N : ", len(Training_N))

        #random.shuffle(Training_N)
        #self.imgs = Training_P+Training_N[:3*len(Training_P)]
        #random.shuffle(self.imgs)
        p = np.random.permutation(len(Training_N))
        self.imgs = Training_P + (np.array(Training_N)[p][:3*len(Training_P)]).tolist()
        self.weights = weights_P + (np.array(weights_N)[p][:3*len(Training_P)]).tolist()

        p = np.random.permutation(4*len(Training_P))
        self.imgs = (np.array(self.imgs)[p]).tolist()
        self.weights = (np.array(self.weights)[p]).tolist()


    def __getitem__(self, idx):
        src_idx, tgt_idx, domain = self.imgs[idx]

        x_src, y_src = self.x_source[src_idx], self.y_source[src_idx]
        x_tgt, y_tgt = self.x_target[tgt_idx], self.y_target[tgt_idx]

        if self.task == "MNIST_to_USPS" or self.task == "USPS_to_MNIST":
            x_src = torch.from_numpy(x_src).unsqueeze(0)
            x_tgt = torch.from_numpy(x_tgt).unsqueeze(0)
        else:
            x_src = torch.from_numpy(x_src)
            x_tgt = torch.from_numpy(x_tgt)

        return x_src, y_src, x_tgt, y_tgt

    def __len__(self):
        return len(self.imgs)


class TestSet(Dataset):
    def __init__(self, domain_adaptation_task, repetition, sample_per_class):
        self.task = domain_adaptation_task
        self.x_test = np.load('./row_data/' + domain_adaptation_task + '_X_test_target_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class)+'.npy')
        self.y_test = np.load('./row_data/' + domain_adaptation_task + '_y_test_target_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class)+'.npy')

    def __getitem__(self, idx):
        x, y = self.x_test[idx], self.y_test[idx]
        if self.task == "MNIST_to_USPS" or self.task == "USPS_to_MNIST":
            x = torch.from_numpy(x).unsqueeze(0)
        else:
            x = torch.from_numpy(x)
        return x, y

    def __len__(self):
        return len(self.x_test)
