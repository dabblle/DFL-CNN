import os
import sys
sys.path.append(os.path.dirname(__file__))

import argparse
from sys import platform
import numpy as np
import random
import time
import cv2 as cv
from models import *  # set ONNX_EXPORT in yolo_yolo_models.py
from utils.datasets import *
from utils.utils import *
from label_convertor import LabelConvertor

DEBUG = True

class LPRecognizer(object):
    def __init__(self, model_path, cfg_path, names_path = 'data/lp.names',img_size=(416, 256), device='0', half=False):
        '''

        :param model_path: yolo weights
        :param cfg_path: darknet config path
        :param img_size:
        :param device: 'cpu' or '1', '0', in infer stage ,only one device
        :param half:
        '''
        if 'cpu' == device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda', int(device))
        # self.device = torch_utils.select_device(device=device)
        self.label_names = self.gen_name_list(names_path)
        self.img_size = img_size
        self.model = Darknet(cfg_path, self.img_size)
        self.weights = model_path
        if self.weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
        else:  # darknet format
            load_darknet_weights(model, self.weights)

        # Eval mode
        self.model.to(self.device).eval()

        # Half precision
        self.half = half and self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()
        self.label_convertor = LabelConvertor(names_path)

    def gen_name_list(self, names_path):
        label_list = []
        with open(names_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            for line in lines:
                label = line.strip()
                label_list.append(label)

        return label_list


    def infer(self, img, conf_thres=0.3, iou_thres=0.5):
        '''

        :param img: opencv format img
        :return:
        '''

        img_ori = img.copy()
        img = self.img_transform(img)

        t0 = time.time()
        with torch.no_grad():
            # infer
            # Get detections
            img = torch.from_numpy(img).to(self.device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = self.model(img)[0]

            if self.half:
                pred = pred.float()

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres)
            #  we only infer one img
            det = pred[0]
            if DEBUG:
                print('det result is {}'.format(det))
            label = None
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_ori.shape).round()
                det = det.cpu().numpy()
                label = self.label_convertor.convert_label(det)
                # print(label)

                # # Write results
                # for *xyxy, conf, cls in det:
                #     label = '%s %.2f' % (self.name, conf)
                #     plot_one_box(xyxy, img_ori, label=label, color=(0,0,255))
                #     cv.imwrite('det_result.jpg', img_ori)

            else:
                print('detect nothing')
        return label

    def img_transform(self, img):
        '''
        :param img: opencv format
        :return:
        '''
        # Padded resize
        img = letterbox(img, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return img


if __name__ == '__main__':
    model_path = '/opt/data/private/code/yolov3/lp_weights/ep1000_bs32_is360_multiscale/best.pt'
    img_path = '/opt/data/private/dataset/lp_Singapore/lp_dataset_label/val/PA5230A.jpg'
    cfg_path = './cfg/yolov3-lp.cfg'
    vehicle_detector = LPRecognizer(model_path, cfg_path, img_size=352)
    img = cv.imread(img_path)
    label = vehicle_detector.infer(img, conf_thres=0.5, iou_thres=0.6)
    print('label: {}'.format(label))



