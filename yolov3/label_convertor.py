import numpy as np
import os

class LabelConvertor(object):
    def __init__(self, class_names_path):
        self.label_list = self.gen_name_list(class_names_path)


    def gen_name_list(self, names_path):
        label_list = []
        with open(names_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            for line in lines:
                label = line.strip()
                label_list.append(label)

        return label_list

    def compute_mean_h(self, bboxes):
        bbox = bboxes[:, :4]
        h = bbox[:, 3] - bbox[:, 1]

        return h.mean()

    def split_two_list(self, bboxes):
        mean_h = self.compute_mean_h(bboxes)

        center_y = (bboxes[:, 1] + bboxes[:, 3])/2

        bottom = []
        upper = []
        for i, y_value in enumerate(center_y):
            if y_value - min(center_y) > 0.6*mean_h:
                bottom.append(i)
            else:
                upper.append(i)

        return upper, bottom

    def sort_by_x(self, bbox):
        idx = np.argsort(bbox[:,0])
        return idx

    def class_id2str(self, label_idx):
        line = ''
        for idx in label_idx:
            line+=self.label_list[int(idx)]
        return line

    def convert_label(self, det_result):

        bboxes =det_result[:, :4]
        upper, bottom = self.split_two_list(bboxes)
        if len(bottom) == 0:
            idx = np.argsort(det_result[:, 0])
            sorted_result = det_result[idx]
            label_idx = sorted_result[:, 5]
            label = self.class_id2str(label_idx)

        else:
            upper = det_result[np.array(upper)]
            bottom = det_result[np.array(bottom)]

            # print(upper)
            # print(bottom)

            upper_sorted = upper[self.sort_by_x(upper)]
            bottom_sorted = bottom[self.sort_by_x(bottom)]

            label_idx_upper = upper_sorted[:, 5]
            label_idx_bottom = bottom_sorted[:, 5]

            upper_str = self.class_id2str(label_idx_upper)
            bottom_str = self.class_id2str(label_idx_bottom)

            label = upper_str+bottom_str

        return label
