import torch
import os
from PIL import Image
from utils.transform import get_transform_for_test_simple
from model.DFL import DFL_VGG16
base_path = '/opt/data/private/DATASETS/CarsDatasets/test/'
def val_top1():
    model = DFL_VGG16(k=10, nclass=176)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('/opt/data/private/DATASETS/CarsDatasets/classnames.name', 'r') as f:
        index2classlist = f.read().split('\n')
    checkpoint = torch.load('./model_best.pth.tar')
    model = torch.nn.DataParallel(model, device_ids=range(0, 1))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    val_transforms = get_transform_for_test_simple()
    for img_floder in os.listdir(base_path):
        right = 0
        for img_name in os.listdir(base_path+img_floder):
            img = Image.open(os.path.join(base_path+img_floder, img_name))
            if len(img.split()) != 3:
                img = img.convert("RGB")
            img_tensor = val_transforms(img)
            img_tensor = img_tensor.unsqueeze(0)
            out1, out2, out3, indices = model(img_tensor)
            out_sum = out1 + out2 + out3 * 0.1
            value, index = torch.max(out_sum.cpu(), 1)
            idx = int(index[0])
            cls_name = index2classlist[idx]
            if img_floder == cls_name:
                right += 1
        print(img_floder, right, len(os.listdir(base_path + img_floder)))
if __name__ == '__main__':
    with torch.no_grad():
        val_top1()