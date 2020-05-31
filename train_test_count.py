import os
# train_path = '/opt/data/private/DATASETS/CarsDatasets/train/'
# test_path = '/opt/data/private/DATASETS/CarsDatasets/test/'

train_path = '/opt/data/private/code/color-DFL-CNN/color_dataset/train/'
test_path = '/opt/data/private/code/color-DFL-CNN/color_dataset/test/'
train_list = os.listdir(train_path)

test_list = os.listdir(test_path)
for image in train_list:
    print(image, ',', len(os.listdir(train_path+image)))