import os
from PIL import Image
import cv2
# color detect

class Colors(object):
    class Color(object):
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return "%s : %s" % (self.__class__.__name__, self.value)

    class Red(Color): pass

    class Blue(Color): pass

    class Green(Color): pass

    class Yellow(Color): pass

    class White(Color): pass

    class Gray(Color): pass

    class Black(Color): pass

    class Pink(Color): pass

    class Teal(Color): pass


class ColorWheel(object):
    def __init__(self, rgb):
        r, g, b = rgb
        self.rgb = (Colors.Red(r), Colors.Green(g), Colors.Blue(b),)

    def estimate_color(self):
        dominant_colors = self.get_dominant_colors()
        total_colors = len(dominant_colors)
        if total_colors == 1:
            return dominant_colors[0]
        elif total_colors == 2:
            color_classes = [x.__class__ for x in dominant_colors]

            if Colors.Red in color_classes and Colors.Green in color_classes:
                return Colors.Yellow(dominant_colors[0].value)
            elif Colors.Red in color_classes and Colors.Blue in color_classes:
                return Colors.Pink(dominant_colors[0].value)
            elif Colors.Blue in color_classes and Colors.Green in color_classes:
                return Colors.Teal(dominant_colors[0].value)
        elif total_colors == 3:
            if dominant_colors[0].value > 200:
                return Colors.White(dominant_colors[0].value)
            elif dominant_colors[0].value > 100:
                return Colors.Gray(dominant_colors[0].value)
            else:
                return Colors.Black(dominant_colors[0].value)
        else:
            print("Dominant Colors : %s" % dominant_colors)

    def get_dominant_colors(self):
        max_color = max([x.value for x in self.rgb])

        # return [x for x in self.rgb if x.value >= max_color * .9]
        return [x for x in self.rgb if x.value >= max_color * .85]


def process_image(image):
    image_color_quantities = {}
    width, height = image.size
    # for x in range(width):
    # for y in range(height):

    width_margin = int(width - (width * .6))
    height_margin = int(height - (height * .6))
    # print height
    # print range(height_margin, height - height_margin)
    for x in range(width_margin, width - width_margin):  #
        for y in range(height_margin, height - height_margin):  #
            r, g, b = image.getpixel((x, y))
            key = "%s:%s:%s" % (r, g, b,)
            key = (r, g, b,)
            image_color_quantities[key] = image_color_quantities.get(key, 0) + 1

    total_assessed_pixels = sum([v for k, v in image_color_quantities.items() if v > 10])

    # strongest_color_wheels = [(ColorWheel(k), v / float(total_pixels) * 100, ) for k, v in test.items() if v > 30]
    strongest_color_wheels = [(ColorWheel(k), v / float(total_assessed_pixels) * 100,) for k, v in
                              image_color_quantities.items() if v > 10]

    final_colors = {}

    for color_wheel, strength in strongest_color_wheels:
        # print "%s => %s" % (strength, [str(x) for x in color_wheel.get_dominant_colors()], )

        # print "%s => %s" % (strength, color_wheel.estimate_color(), )

        color = color_wheel.estimate_color()

        final_colors[color.__class__] = final_colors.get(color.__class__, 0) + strength
    max = 0
    recolor = ''
    for color, strength in final_colors.items():
        if max < strength:
            max = strength
            recolor = color
    # print ("%s - %s" % (recolor.__name__, max, ))
    if recolor =='':
        return None
    return recolor.__name__

############ another color detect ###############
import numpy as np
import collections

def getColorList():
    dict = collections.defaultdict(list)

    # 黑色
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])
    color_list = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    dict['black'] = color_list

    #灰色
    lower_gray = np.array([0, 0, 46])
    upper_gray = np.array([180, 43, 220])
    color_list = []
    color_list.append(lower_gray)
    color_list.append(upper_gray)
    dict['gray']=color_list

    # 白色
    lower_white = np.array([0, 0, 221])
    upper_white = np.array([180, 30, 255])
    color_list = []
    color_list.append(lower_white)
    color_list.append(upper_white)
    dict['white'] = color_list

    # 红色
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red'] = color_list

    # 红色2
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red2'] = color_list

    # 橙色
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([25, 255, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    dict['orange'] = color_list

    # 黄色
    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list

    # 绿色
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list

    # 青色
    lower_cyan = np.array([78, 43, 46])
    upper_cyan = np.array([99, 255, 255])
    color_list = []
    color_list.append(lower_cyan)
    color_list.append(upper_cyan)
    dict['cyan'] = color_list

    # 蓝色
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    dict['blue'] = color_list

    # 紫色
    lower_purple = np.array([125, 43, 46])
    upper_purple = np.array([155, 255, 255])
    color_list = []
    color_list.append(lower_purple)
    color_list.append(upper_purple)
    dict['purple'] = color_list

    return dict
# 处理图片
def get_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maxsum = -100
    color = None
    color_dict = getColorList()
    for d in color_dict:
        mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
        cv2.imwrite(d + '.jpg', mask)
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary, None, iterations=2)
        img, cnts, hiera = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sum = 0
        for c in cnts:
            sum += cv2.contourArea(c)
        if sum > maxsum:
            maxsum = sum
            color = d

    return color
save_path = 'C:/Users/HUAWEI/Desktop/result/'
if __name__ == '__main__':
    path = 'C:/Users/HUAWEI/Desktop/test/'
    for img in os.listdir(path):
        image = Image.open(os.path.join(path, img))
        color = process_image(image)
        image.save(os.path.join(save_path, color + img))
