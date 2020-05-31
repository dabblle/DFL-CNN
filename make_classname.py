import os
import shutil
base_path = '/opt/data/private/DATASETS/CarsDatasets/'
dec_path = '/opt/data/private/code/DFL-CNN/Make_dataset/'
org_train_dir = os.path.join(base_path, 'train')
org_test_dir = os.path.join(base_path, 'test')

dec_train_dir = os.path.join(dec_path, 'train')
dec_test_dir = os.path.join(dec_path, 'test')
if not os.path.exists(dec_test_dir):
    os.mkdir(dec_test_dir)
if not os.path.exists(dec_train_dir):
    os.mkdir(dec_train_dir)
org_train_list = os.listdir(org_train_dir)
org_test_list = os.listdir(org_test_dir)
for cls in org_train_list:
    cls_dir = os.path.join(dec_train_dir, cls.split('_')[0])
    if not os.path.exists(cls_dir):
        os.mkdir(cls_dir)
    for img in os.listdir(os.path.join(org_train_dir, cls)):
        src = os.path.join(os.path.join(org_train_dir, cls), img)
        dec = os.path.join(cls_dir, img)
        shutil.copy(src, dec)
for cls in org_test_list:
    cls_dir = os.path.join(dec_test_dir, cls.split('_')[0])
    if not os.path.exists(cls_dir):
        os.mkdir(cls_dir)
    for img in os.listdir(os.path.join(org_test_dir, cls)):
        src = os.path.join(os.path.join(org_test_dir, cls), img)
        dec = os.path.join(cls_dir, img)
        shutil.copy(src, dec)
print('train')
for cls in os.listdir(dec_path+'train/'):
    print(cls, '---->', len(os.listdir(dec_path+'train/'+cls)))
print('test')
for cls in os.listdir(dec_path+'test/'):
    print(cls, '---->', len(os.listdir(dec_path+'test/'+cls)))