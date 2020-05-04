
import torch.nn.parallel
from process_data import process_data
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.MyImageFolderWithPaths import *
from drawrect import *

def get_transform():
    transform_list = []

    transform_list.append(transforms.Lambda(lambda img: scale_keep_ar_min_fixed(img, 448)))

    transform_list.append(transforms.Resize((448, 448)))

    transform_list.append(transforms.ToTensor())

    transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    return transforms.Compose(transform_list)

def fine_grained():

    with open('/opt/data/private/DATASETS/CarsDatasets/classnames.name', 'r') as f:
        index2classlist = f.read().split('\n')
    checkpoint=torch.load('weight/model_best.pth.tar')
    model = DFL_VGG16(k=10, nclass=175)
    model = nn.DataParallel(model, device_ids=range(0,1))
    model = model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    base_path = '/opt/data/private/code/DFL-CNN/vis_img/'
    save_path = '/opt/data/private/code/DFL-CNN/tmp/'
    i = 0
    # for img in os.listdir(base_path):
    #     image = Image.open(base_path + img)
    #     resize_image = process_data(image)
    #     img_tensor = get_transform()(resize_image)
    #     img_tensor = img_tensor.unsqueeze(0)
    #     out1, out2, out3, indices = model(img_tensor)
    #     out = out1 + out2 + 0.1 * out3
    #     value, index = torch.max(out.cpu(), 1)
    #     idx = int(index[0])
    #     dirname = index2classlist[idx]
    #     print('iamge name:{0},predict class:{1}'.format(img, dirname))
    #     image.save(save_path + dirname+'_%s'% i+'.jpg')
    #     i += 1




if __name__ == '__main__':
    with torch.no_grad():
        fine_grained()
