from PIL import Image
import numpy as np
import os
from torchvision import transforms as T


transform = T.Resize(224)
root = '../domain_adaptation_images/'
sub_dirs = os.listdir(root)
c=0
for i in sub_dirs:
    if i == '.DS_Store':
        sub_dirs.remove(i)
domain = ['amazon','dslr','webcam']
domain_path = [os.path.join(root,sub_dir) for sub_dir in sub_dirs]
for each_domain_path in domain_path:
    images = os.path.join(each_domain_path,'images')
    catogories = os.listdir(images)
    for i in catogories:
        if i == '.DS_Store':
            catogories.remove(i)
    catogories_path = [os.path.join(images,catogory) for catogory in catogories]
    label = 0
    for catogory_path in catogories_path:
        arr = []
        images_path = os.listdir(catogory_path)
        for i in images_path:
            if i == '.DS_Store':
                images_path.remove(i)
        img_path = [os.path.join(catogory_path,image_path) for image_path in images_path]
        for each_img_path in img_path:
            img = Image.open(each_img_path)
            img = transform(img)
            arr1 = np.array(img)
            arr1 = np.transpose(arr1,(2,0,1))
            arr1 = np.expand_dims(arr1,axis=0)
            arr.append(arr1)
        res = arr[0]

        for j in range(1,len(arr)):
            res = np.concatenate((res,arr[j]))
        print(res.shape)
        print(c)
        domain_name = domain[c]
        filename = f"{domain_name}_{label}.npy"
        np.save(filename,res)
        label += 1
    c += 1






# raw pic 300*300*3
# path = '../domain_adaptation_images/amazon/images/back_pack'
# imgpath = os.path.join(path,'frame_0001.jpg')
# img = Image.open(imgpath)
# arr = np.array(img)
# print(arr.shape)
# np.save("office31.npy", arr)