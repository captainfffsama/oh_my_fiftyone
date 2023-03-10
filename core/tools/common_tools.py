# -*- coding: utf-8 -*-
"""
@Author: captainsama
@Date: 2023-03-08 15:10:57
@LastEditors: captainsama tuanzhangsama@outlook.com
@LastEditTime: 2023-03-08 15:18:15
@FilePath: /dataset_manager/core/tools/common_tools.py
@Description:
"""

from typing import Optional, Union, List
from pprint import pprint
from datetime import datetime
from functools import wraps

import os
import json
from concurrent import futures

import fiftyone as fo
import fiftyone.core.dataset as focd
from tqdm import tqdm
from PIL import Image
import piexif
import cv2

from core.utils import get_sample_field, md5sum, get_all_file_path
from core.exporter.sgccgame_dataset_exporter import SGCCGameDatasetExporter
from core.logging import logging

from core.cache import WEAK_CACHE
from core.model.object_detection import ProtoBaseDetection, ChiebotObjectDetection


def print_time_deco(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print("操作完成时间: {}".format(datetime.now()))
        return result

    return wrapper


@print_time_deco
def get_select_dv(txt_path: str = None) -> Optional[fo.DatasetView]:
    """返回被选中的数据的视图,若有txt就返回txt中的,没有就是浏览器中选中的
    Args:
        txt_path (Optional[str]):txt是一个记录了图片路径的文本文件

    Returns:
        Optional[fo.DatasetView]: 返回被选中的数据的视图
    """

    session = WEAK_CACHE.get("session", None)
    if session is None:
        logging.warning("no dataset in cache,no thing export")
        return
    else:
        dataset = session.dataset
    if dataset and session:
        if txt_path is not None:
            if os.path.exists(txt_path):
                imgs_path = get_all_file_path(txt_path)
                return dataset.select_by("filepath", imgs_path)
        else:
            return dataset.select(session.selected)
    return None


@print_time_deco
def dataset_value2txt(
    value: str = "filepath",
    save_txt: Optional[str] = None,
    dataset: Optional[focd.Dataset] = None,
):
    """将数据集的特定字段导入到txt

    Args:
        value (str, optional): 需要导出的字段. Defaults to "filepath".
        save_txt (Optional[str], optional): 需要保存的txt的路径,若为None就仅print出. Defaults to None.
        dataset (Optional[focd.Dataset], optional): 需要导出字段的数据集,若为None就是 ``session.dataset``. Defaults to None.
    """
    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            dataset = s.dataset

    v = dataset.values(value)
    if save_txt:
        with open(save_txt, "w") as fw:
            for vv in v:
                fw.write(str(vv) + "\n")
    else:
        pprint(v)


@print_time_deco
def imgslist2dataview(
    imgslist: Union[str, List[str]], dataset: Optional[focd.Dataset] = None
) -> fo.DatasetView:
    """传入文件列表本身或者路径得到对应的dataview

    Args:
        imgslist (Union[str, List[str]]): 可以是一个记录了图片文件绝对路径的txt,或者图片目录或者是一个pythonlist
        dataset (Optional[focd.Dataset], optional): 同以前函数. Defaults to None.

    Returns:
        fo.DatasetView: 图片list的dataview
    """
    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            dataset = s.dataset

    if isinstance(imgslist, str):
        imgslist = get_all_file_path(imgslist)

    return dataset.select_by("filepath", imgslist)


@print_time_deco
def check_dataset_exif(
    dataset: Optional[focd.Dataset] = None,
    clean_inplace: bool = False,
    log_path: Optional[str] = None,
    cv2_fix: bool = False,
) -> fo.DatasetView:
    """检查数据集中是否包含exif

    Args:
        dataset (Optional[focd.Dataset], optional): _description_. Defaults to None.
        clean_inplace (bool, optional): 是否原地移除exif. Defaults to False.
        log_path (Optional[str], optional): 若不为None,则将包含了exif的样本路径导出到txt. Defaults to None.
        cv2_fix (bool, optinal): 在clean_inplace为True的情况下,若piexif报错,指示是否使用opencv读入读出. Defaults False,
    Returns:
        fo.DatasetView: 包含了exif的数据的dataview
    """
    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            dataset = s.dataset

    have_exif = []
    for sample in tqdm(
        dataset,
        desc="exif 检查进度:",
        dynamic_ncols=True,
        colour="green",
    ):
        img = Image.open(sample["filepath"])
        if "exif" in img.info:
            have_exif.append(sample["filepath"])
            if clean_inplace:
                try:
                    piexif.remove(sample["filepath"])
                except Exception as e:
                    logging.critical(
                        "{} remove piexif faild".format(sample["filepath"])
                    )
                    print("{} remove piexif faild".format(sample["filepath"]))
                    if cv2_fix and isinstance(e, piexif.InvalidImageDataError):
                        img = cv2.imread(
                            sample.filepath,
                            cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR,
                        )
                        cv2.imwrite(sample.filepath, img)

    if log_path:
        with open(log_path, "w") as fw:
            fw.writelines([x + "\n" for x in have_exif])

    print("Here is exif image path:")
    pprint(have_exif)

    return imgslist2dataview(have_exif, dataset)


@print_time_deco
def model_det(
    dataset: Optional[focd.Dataset] = None,
    model: Optional[ProtoBaseDetection] = None,
    model_initargs: Optional[dict] = None,
):
    """使用模型检测数据集,并将结果存到sample的model_predic 字段

    Args:
        dataset (Optional[focd.Dataset], optional): 同之前. Defaults to None.
        model (Optional[ProtoBaseDetection], optional): 用于检测模型实例. Defaults to None.默认使用ChiebotObjectDetection
        model_initargs: (Optional[dict],optinal): 用于初始化默认模型实例的参数,对于ChiebotObjectDetection就是模型类型
    """
    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            dataset = s.dataset
    if model is None:
        if model_initargs is None:
            model_initargs = {}
        model = ChiebotObjectDetection(**model_initargs)

    if isinstance(model, ProtoBaseDetection):
        with fo.ProgressBar(
            total=len(dataset), start_msg="模型检测进度:", complete_msg="检测完毕"
        ) as pb:
            with model as m:
                deal_one = lambda s, mm: (s, mm(s.filepath))
                with futures.ThreadPoolExecutor(10) as exec:
                    tasks = [
                        exec.submit(deal_one, sample, m) for sample in dataset
                    ]
                    for task in pb(futures.as_completed(tasks)):
                        sample, objs = task.result()
                        det_r = [
                            fo.Detection(
                                label=obj[0], bounding_box=obj[2:], confidence=obj[1]
                            )
                            for obj in objs
                        ]

                        sample["model_predict"] = fo.Detections(
                            detections=det_r
                        )

                        sample.save()
    else:
        pass

    session = WEAK_CACHE.get("session", None)
    if session is not None:
        session.refresh()