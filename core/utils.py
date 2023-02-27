import hashlib
from typing import Tuple
import time
from contextlib import contextmanager
import os
import xml.etree.ElementTree as ET

import fiftyone.core.metadata as fom
from PIL import Image
import numpy as np


def get_all_file_path(file_dir: str, filter_=('.jpg')) -> list:
    #遍历文件夹下所有的file
    if os.path.isdir(file_dir):
        return [os.path.join(maindir,filename) for maindir,_,file_name_list in os.walk(file_dir) \
            for filename in file_name_list \
            if os.path.splitext(filename)[1] in filter_ ]
    elif os.path.isfile(file_dir):
        with open(file_dir, 'r') as fr:
            paths = [x.strip() for x in fr.readlines()]
        return paths
    else:
        raise ValueError("{} should be dir or a txt file".format(file_dir))


PIL_MODE_CHANNEL_MAP = {
    "1": 1,
    "L": 1,
    "P": 1,
    "RGB": 3,
    "RGBA": 4,
    "CMYK": 4,
    "YCbCr": 3,
    "LAB": 3,
    "HSV": 3,
    "I": 1,
    "F": 1,
    "LA": 2,
    "PA": 2,
    "RGBX": 3,
    "RGBa": 4,
    "La": 2,
    "I;16": 1,
    "I;16L": 1,
    "I;16B": 1,
    "I;16N": 1,
    "BGR;15": 3,
    "BGR;16": 3,
    "BGR;24": 3,
    "BGR;32": 3,
}


def parse_xml_info(xml_path):
    ''' 解析xml文件信息
    解析出的xml信息包含2类：
    第一类是图像信息：图像名图像宽高,通道数
    第二类是包含的目标信息：目标类别和每类目标所有bbx的位置
    Args:
        xml_path:xml文件路径
    Return
        img_info: [list], [img_name, W, H, C]
        obj_info: [dict], {obj_name1: [[xmin,ymin,xmax,ymax], [xmin,ymin,xmax,ymax], ...], obj_name2: ...}
    '''
    assert os.path.exists(xml_path), "{0} does not exist!".format(xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_name = root.find('filename').text
    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)
    img_depth = int(root.find('size/depth').text)
    img_info = [img_name, img_width, img_height, img_depth]

    obj_info = {}
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        if obj_name not in obj_info.keys():
            obj_info[obj_name] = []
        obj_info[obj_name].append((xmin, ymin, xmax, ymax))

    return img_info, obj_info


def parse_img_metadata(img_path) -> fom.ImageMetadata:
    img = Image.open(img_path)
    return fom.ImageMetadata(mime_type=img.format,
                             width=img.width,
                             height=img.height,
                             num_channels=PIL_MODE_CHANNEL_MAP.get(
                                 img.mode, "3"),
                             img_path=img_path)


def normalization_xyxy(
        xyxy: tuple, w: int,
        h: int) -> Tuple[Tuple[float, float, float, float], bool]:
    """将 xmin,ymin,xmax,ymax 转化成 tlx,tly,w,h,数值归一化到[0,1]

    Args:
        xyxy (tuple): xmin,ymin,xmax,ymax
        w (int): 图片宽
        h (int): 图片高

    Returns:
        Tuple[Tuple[float,float,float,float],bool]: 前者是 (tlx,tly,w,h),后者是指示是否有目标超出图片大小
    """
    flag = True
    xmin, ymin, xmax, ymax = xyxy
    if xmax <= xmin:
        xmin, xmax = xmax, xmin
        flag = False

    if ymax <= ymin:
        ymin, ymax = ymax, ymin
        flag = False

    if not (0 <= xmax <= w):
        xmax = int(np.clip(xmax, 0, w))
        flag = False

    if not (0 <= xmin <= w):
        xmin = int(np.clip(xmin, 0, w))
        flag = False

    if not (0 <= ymax <= h):
        ymax = int(np.clip(ymax, 0, h))
        flag = False

    if not (0 <= ymin <= h):
        ymin = int(np.clip(ymin, 0, h))
        flag = False

    return (xmin/w,ymin/h,(xmax-xmin)/w,(ymax-ymin)/h),flag

@contextmanager
def timeblock(label:str = '\033[1;34mSpend time:\033[0m'):
    r'''上下文管理测试代码块运行时间,需要
        import time
        from contextlib import contextmanager
    '''
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print('\033[1;34m{} : {}\033[0m'.format(label, end - start))


def md5sum(count_str:str) -> str:
    m = hashlib.md5()
    if os.path.isfile(count_str):
        with open(count_str,'rb') as frb:
            m.update(frb.read())
    else:
        m.update(count_str.encode('utf-8'))
    return m.hexdigest()
