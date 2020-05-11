from torchvision import datasets, transforms, utils
import torch.utils.data
import os
from drawrect import *
dataroot = '/opt/data/private/DATASETS/CarsDatasets/'
traindir = os.path.join(dataroot, 'train')
def get_transform_for_train():
    transform_list = []

    transform_list.append(transforms.Lambda(lambda img: scale_keep_ar_min_fixed(img, 448)))

    transform_list.append(transforms.RandomHorizontalFlip(p=0.3))

    transform_list.append(transforms.Resize((448, 448)))

    transform_list.append(transforms.ToTensor())

    transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    return transforms.Compose(transform_list)
transform_train = get_transform_for_train()
train_dataset = ImageFolderWithPaths(traindir, transform = transform_train)
index2classlist = train_dataset.index2classlist()
for index in index2classlist:
    print(index)