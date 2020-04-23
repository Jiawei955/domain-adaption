import os
import h5py
# import cv2
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms as T

def preprocess_office(in_path, out_path):
    """

    Args:
        in_path:
        out_path: file name the data will be saved

    Returns:

    """
    transform = T.Resize(224)
    cat_list = sorted(os.listdir(in_path))
    label = 0
    lst = []  # store numpy arrays of images in AMAZON
    labels = []
    for i in cat_list:
        if i[0]=='.':
            continue

        cat_path = os.path.join(in_path,i)
        img_list = os.listdir(cat_path)
        for j in img_list:
            if j[0]=='.':
                continue
            img_path = os.path.join(cat_path,j)
            img = transform(Image.open(img_path))
            img_arr = np.array(img)
            img_arr = np.transpose(img_arr,(2,0,1))
            # img_arr = np.expand_dims(img_arr,axis=0)
            lst.append(img_arr)
            labels.append(label)
        label += 1

    data = np.asarray(lst)
    target = np.asarray(labels)
    print(data.shape,target.shape)
    np.savez(out_path, data=data, target=target)

    # print(data.shape)
    # plt.imshow(data[0])
    # plt.show()
    # print(target[0])
    # np.savez(out_path, data=data, target=target)

def preprocess_usps(in_path, out_path):
    hf = h5py.File(in_path, 'r')
    train = hf['train']
    test = hf['test']

    data = np.concatenate([np.array(train['data']), np.array(test['data'])])
    target = np.concatenate([np.array(train['target']), np.array(test['target'])])
    data = data.reshape([-1, 16, 16])
    print(data.shape)
    # plt.imshow(data[0])
    # plt.show()
    # print(target[0])
    np.savez(out_path, data=data, target=target)

def preprocess_mnist(in_paths, out_path):
    train_dataset = pd.read_csv(in_paths[0]).values
    test_dataset = pd.read_csv(in_paths[1]).values

    data = np.concatenate([train_dataset[:, 1:], test_dataset[:, 1:]])
    target = np.concatenate([train_dataset[:, 0], test_dataset[:, 0]])
    data = data.reshape([-1, 28, 28]).astype(np.uint8)
    new_data = np.zeros((data.shape[0], 16, 16))

    for i in range(data.shape[0]):
        new_data[i] = cv2.resize(data[i], (16,16))
    data = new_data/255
    print(data.shape)
    # plt.imshow(data[0])
    # plt.show()
    # print(target[0])
    np.savez(out_path, data=data, target=target)

if __name__ == '__main__':
    # preprocess_usps('../torchvision_datasets/USPS/usps.h5', '../torchvision_datasets/USPS/USPS')
    # preprocess_mnist(['../torchvision_datasets/MNIST/mnist_train.csv', '../torchvision_datasets/MNIST/mnist_test.csv'], '../torchvision_datasets/MNIST/MNIST')
    preprocess_office('./office31/amazon/images', './office31/amazon/AMAZON')
    preprocess_office('./office31/dslr/images', './office31/dslr/DSLR')
    preprocess_office('./office31/webcam/images', './office31/webcam/WEBCAM')