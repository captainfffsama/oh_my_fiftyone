import base64
import hashlib
from typing import Tuple, Union, Dict, List, Optional
import time
from contextlib import contextmanager
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import json
import shutil
from datetime import datetime
from tqdm import tqdm

import numpy as np
import cv2
import fiftyone as fo
import fiftyone.core.metadata as fom
import fiftyone.core.labels as fol
import fiftyone.core.view as focv
from PIL import Image
import requests
from sklearn.model_selection import train_test_split
from core.logging import logging


def optimize_view(
    dataset: Union[fo.Dataset, fo.DatasetView]
) -> Union[fo.Dataset, fo.DatasetView]:
    if isinstance(dataset, focv.DatasetView):
        return focv.make_optimized_select_view(dataset, dataset.values("id"))
    else:
        return dataset


def get_all_file_path(
    file_dir: str,
    filter_=(".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP", ".jpeg", ".JPEG"),
) -> list:
    # 遍历文件夹下所有的file
    if os.path.isdir(file_dir):
        return [
            os.path.join(maindir, filename)
            for maindir, _, file_name_list in os.walk(file_dir)
            for filename in file_name_list
            if os.path.splitext(filename)[1] in filter_
        ]
    elif os.path.isfile(file_dir):
        with open(file_dir, "r") as fr:
            paths = [
                os.path.abspath(x.strip())
                for x in fr.readlines()
                if os.path.splitext(x.strip())[1] in filter_
            ]
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
    """解析xml文件信息
    解析出的xml信息包含2类：
    第一类是图像信息：图像名图像宽高,通道数
    第二类是包含的目标信息：目标类别和每类目标所有bbx的位置
    Args:
        xml_path:xml文件路径
    Return
        img_info: [list], [img_name, W, H, C]
        obj_info: [dict], {obj_name1: [[xmin,ymin,xmax,ymax], [xmin,ymin,xmax,ymax], ...], obj_name2: ...}
    """
    assert os.path.exists(xml_path), "{0} does not exist!".format(xml_path)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        img_name = root.find("filename").text
        img_width = int(root.find("size/width").text)
        img_height = int(root.find("size/height").text)
        img_depth = int(root.find("size/depth").text)
        img_info = [img_name, img_width, img_height, img_depth]

        obj_info = {}
        for obj in root.findall("object"):
            obj_name = obj.find("name").text
            xmin = int(float(obj.find("bndbox/xmin").text))
            ymin = int(float(obj.find("bndbox/ymin").text))
            xmax = int(float(obj.find("bndbox/xmax").text))
            ymax = int(float(obj.find("bndbox/ymax").text))

            if obj_name not in obj_info.keys():
                obj_info[obj_name] = []
            obj_info[obj_name].append((xmin, ymin, xmax, ymax))
    except Exception as e:
        logging.critical("{} xml is wrong".format(xml_path))
        print("{} is wrong".format(xml_path))
        raise e

    return img_info, obj_info


def parse_img_metadata(img_path) -> fom.ImageMetadata:
    img = Image.open(img_path)
    return fom.ImageMetadata(
        size_bytes=os.path.getsize(img_path),
        mime_type=img.format,
        width=img.width,
        height=img.height,
        num_channels=PIL_MODE_CHANNEL_MAP.get(img.mode, "3"),
    )


def normalization_xyxy(
    xyxy: tuple, w: int, h: int
) -> Tuple[Tuple[float, float, float, float], bool]:
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

    return (xmin / w, ymin / h, (xmax - xmin) / w, (ymax - ymin) / h), flag


@contextmanager
def timeblock(label: str = "\033[1;34mSpend time:\033[0m"):
    r"""上下文管理测试代码块运行时间,需要
    import time
    from contextlib import contextmanager
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print("\033[1;34m{} : {}\033[0m".format(label, end - start))


def md5sum(count_str: str) -> str:
    m = hashlib.md5()
    if os.path.isfile(count_str):
        with open(count_str, "rb") as frb:
            m.update(frb.read())
    else:
        m.update(count_str.encode("utf-8"))
    return m.hexdigest()


def get_sample_field(sample, field, default=None):
    if sample.has_field(field):
        return sample.get_field(field)
    else:
        return default


def img2base64(file: Union[str, np.ndarray]) -> bytes:
    if isinstance(file, str):
        img_file = open(file, "rb")  # 二进制打开图片文件
        img_b64encode = base64.b64encode(img_file.read())  # base64编码
        img_file.close()  # 文件关闭
        return img_b64encode
    elif isinstance(file, np.ndarray):
        img_str = cv2.imencode(".jpg", file)[
            1
        ].tostring()  # 将图片编码成流数据，放到内存缓存中，然后转化成string格式
        img_b64encode = base64.b64encode(img_str)  # 编码成base64
        return img_b64encode
    else:
        return None


def base642img(base64code: bytes) -> np.ndarray:
    str_decode = base64.b64decode(base64code)
    nparr = np.fromstring(str_decode, np.uint8)
    img_restore = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_restore


def tensor_proto2np(tensor_pb):
    np_matrix = np.array(tensor_pb.data, dtype=np.float).reshape(tensor_pb.shape)
    return np_matrix


def NMS(boxes, iou_thr=0.5, sort_by="area"):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    if sort_by == "area":
        # small to large
        idxs = np.argsort(area)[::-1]
    else:
        idxs = np.argsort(boxes[:, 4])
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        iou = (w * h) / (area[idxs[:last]] + area[i] - (w * h))
        idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > iou_thr)[0])))
    return boxes[pick]


def det_labels2npdict(
    labels: Union[List[fol.Detection], fol.Detections]
) -> Dict[str, np.ndarray]:
    if isinstance(labels, fol.Detections):
        labels = labels.detections

    result = defaultdict(list)
    for label in labels:
        x1, y1, w, h = label.bounding_box
        score = label.confidence if label.confidence is not None else 1.0
        cls_name = label.label
        result[cls_name].append([x1, y1, x1 + w, y1 + h, score])

    for k, v in result.items():
        result[k] = np.array(v)

    return result


def np2dict2det_labels(label_dict: Dict[str, np.ndarray]) -> list:
    result_list = []
    for k, v in label_dict.items():
        for obj in v:
            x1, y1, x2, y2, s = obj.tolist()
            result_list.append(
                fol.Detection(label=k, bounding_box=[x1, y1, x2 - x1, y2 - y1])
            )

    return result_list


def fol_det_nms(
    labels: Union[List[fol.Detection], fol.Detections], iou_thr=0.5, sort_by="area"
):
    npdict = det_labels2npdict(labels)

    r = []
    for k, v in npdict.items():
        nms_r = NMS(v, iou_thr, sort_by)
        for obj in nms_r:
            x1, y1, x2, y2, s = obj.tolist()
            r.append(fol.Detection(label=k, bounding_box=[x1, y1, x2 - x1, y2 - y1]))

    return fol.Detections(detections=r)


def _export_one_sample_anno(
    sample, save_dir, backup_dir=None, export_classes: Optional[list] = None
):
    result = {}
    need_export_map = {
        "data_source": "data_source",
        "img_quality": "img_quality",
        "additions": "additions",
        "tags": "sample_tags",
        "chiebot_ID": "ID",
    }

    for k, v in need_export_map.items():
        vv = get_sample_field(sample, k)
        if vv:
            result[v] = vv

    result["chiebot_sample_tags"] = get_sample_field(
        sample, "chiebot_sample_tags", default=[]
    )

    result["img_shape"] = (
        sample["metadata"].height,
        sample["metadata"].width,
        sample["metadata"].num_channels,
    )
    result["objs_info"] = []
    dets = get_sample_field(sample, "ground_truth")
    if dets:
        for det in dets.detections:
            if export_classes is not None and det.label not in export_classes:
                continue
            obj = {}
            obj["name"] = det.label
            obj["pose"] = "Unspecified"
            obj["truncated"] = 0
            obj["difficult"] = 0
            obj["mask"] = []
            obj["confidence"] = -1
            obj["quality"] = 10
            obj["bbox"] = (
                det.bounding_box[0],
                det.bounding_box[1],
                det.bounding_box[0] + det.bounding_box[2],
                det.bounding_box[1] + det.bounding_box[3],
            )

            result["objs_info"].append(obj)

    embedding: Optional[np.ndarray] = get_sample_field(sample, "embedding", None)

    if embedding is not None:
        result["embedding"] = base64.b64encode(embedding.tobytes()).decode("utf-8")

    save_path = os.path.join(save_dir, os.path.splitext(sample.filename)[0] + ".anno")

    if backup_dir is not None:
        ori_anno = os.path.splitext(sample.filepath)[0] + ".anno"
        if os.path.exists(ori_anno):
            shutil.copy(
                ori_anno,
                os.path.join(
                    backup_dir, os.path.splitext(sample.filename)[0] + ".anno"
                ),
            )

    with open(save_path, "w") as fw:
        json.dump(result, fw, indent=4, sort_keys=True)

    return save_path


def _export_one_sample(
    sample,
    exporter,
    get_anno: bool,
    save_dir,
    export_classes: Optional[list] = None,
    label_filed: str = "ground_truth",
):
    image_path = sample.filepath

    metadata = sample.metadata
    if exporter.requires_image_metadata and metadata is None:
        metadata = fo.ImageMetadata.build_for(image_path)

    # Assumes single label field case
    if export_classes is None:
        label = sample[label_filed]
    else:
        label = []
        for obj in sample[label_filed].detections:
            if obj.label in export_classes:
                label.append(obj)

        label = fol.Detections(detections=label)

    exporter.export_sample(image_path, label, metadata=metadata)

    if get_anno:
        _export_one_sample_anno(sample, save_dir, export_classes=export_classes)


def return_now_time():
    now_time = datetime.now()
    return "{}_{}_{}_{}_{}_{}".format(
        now_time.year,
        now_time.month,
        now_time.day,
        now_time.hour,
        now_time.minute,
        now_time.second,
    )


def get_latest_version(repo_owner, repo_name):
    try:
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/latest"
        response = requests.get(url)
        if response.status_code == 200:
            latest_release = response.json()
            version = latest_release["tag_name"]
            return version
        else:
            return None
    except Exception as e:
        return None


def analytics_split(dataset, class_list):
    cls_info = defaultdict(int)
    for sample in dataset:
        detections = sample["ground_truth"]["detections"]
        cur_list = []
        for det in detections:
            single_cls = det['label']
            if single_cls in class_list:
                cls_info[single_cls] += 1
    return cls_info


def split_data_force(filepath, split_ratio):
    temp_test, temp_val = [], []
    try:
        temp_train, temp_test = train_test_split(
            filepath, train_size=split_ratio[0], test_size=round(1 - split_ratio[0], 5)
        )
        if len(split_ratio) == 3:
            temp_val, temp_test = train_test_split(
                temp_test,
                train_size=round(split_ratio[1] / (1 - split_ratio[0]), 5),
                test_size=round(split_ratio[2] / (1 - split_ratio[0]), 5),
            )
        assert len(temp_train) + len(temp_val) + len(temp_test) == len(
            filepath
        ), "数据太少划分失败"
    except Exception as e:
        if len(filepath) <= 6:
            temp_test, temp_val = [], []
            temp_train = filepath
            return (temp_train, temp_val, temp_test)
        else:
            raise ValueError

    return (temp_train, temp_val, temp_test)
