from model.DFL import DFL_VGG16
from utils.util import *
from utils.transform import *
from train import *
from validate import *
from utils.init import *
import sys
import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.MyImageFolderWithPaths import *
from drawrect import *
from process_data import process_data
from yolov3.colorDetector import *
# yolo
import argparse
from sys import platform
from yolov3.models import *  # set ONNX_EXPORT in models.py
from yolov3.utils.datasets import *
from yolov3.utils.utils import *
vehicle_id = [2, 3, 5, 7]


def get_transform():
    transform_list = []

    transform_list.append(transforms.Lambda(lambda img: scale_keep_ar_min_fixed(img, 448)))

    # transform_list.append(transforms.RandomHorizontalFlip(p=0.3))

    # transform_list.append(transforms.CenterCrop((448, 448)))

    transform_list.append(transforms.ToTensor())

    transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    return transforms.Compose(transform_list)


def detect(save_img=False):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights = opt.output, opt.source, opt.weights

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)

    # Initialize model
    model = Darknet(opt.cfg, img_size)
    fine_grained_model = DFL_VGG16(k=10, nclass=176)
    with open('/opt/data/private/DATASETS/CarsDatasets/classnames.name', 'r') as f:
        index2classlist = f.read().split('\n')
    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    model.to(device).eval()

    checkpoint = torch.load('weight/model_best.pth.tar')
    fine_grained_model = nn.DataParallel(fine_grained_model, device_ids=range(0, 1))
    fine_grained_model.load_state_dict(checkpoint['state_dict'])
    fine_grained_model.to(device).eval()

    dataset = LoadImages(source, img_size=img_size)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            exist_vehicle = False

            p, s, im0 = path, '', im0s
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    # exist car
                    if c in vehicle_id:
                        exist_vehicle = True
                        save_exist_vehicle_img_path = str(Path(out) / Path(p).name)

                # Write results
                for *xyxy, conf, cls in det:
                    if exist_vehicle:
                        h, w, _ = im0.shape
                        (x1, y1) = (int(xyxy[1]), int(xyxy[0]))
                        (x2, y2) = (int(xyxy[3]), int(xyxy[2]))
                        if (x2 - x1)*(y2 - y1)<(h*w*0.1):
                            continue
                        car_image = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                        h, w, _ = car_image.shape
                        pil_image = Image.fromarray(cv2.cvtColor(car_image, cv2.COLOR_BGR2RGB))

                        img_tensor = get_transform()(process_data(pil_image))
                        img_tensor = img_tensor.unsqueeze(0)
                        out1, out2, out3, indices = fine_grained_model(img_tensor)
                        out_sum = out1 + out2 + out3 * 0.1
                        value, index = torch.max(out_sum.cpu(), 1)
                        idx = int(index[0])
                        cls_name = index2classlist[idx]

                        color_image = get_color(car_image[int(h*0.15):h-int(h*0.15), int(w*0.1):w-int(w*0.1)])

                        label = '%s,color:%s' % (cls_name, color_image)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                cv2.imwrite(save_exist_vehicle_img_path, im0)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov3/cfg/yolov3.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='yolov3/data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='yolov3/weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='./vis_img', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='./output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
