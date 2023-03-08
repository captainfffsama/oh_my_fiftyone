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

import os
import json
from concurrent import futures

import fiftyone as fo
import fiftyone.core.dataset as focd
from tqdm import tqdm
from PIL import Image
import piexif

from core.utils import get_sample_field, md5sum, get_all_file_path
from core.exporter.sgccgame_dataset_exporter import SGCCGameDatasetExporter
from core.logging import logging

from core.cache import WEAK_CACHE


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


def check_dataset_exif(
    dataset: Optional[focd.Dataset] = None,
    clean_inplace: bool = False,
    log_path: Optional[str] = None,
) -> fo.DatasetView:
    """检查数据集中是否包含exif

    Args:
        dataset (Optional[focd.Dataset], optional): _description_. Defaults to None.
        clean_inplace (bool, optional): 是否原地移除exif. Defaults to False.
        log_path (Optional[str], optional): 若不为None,则将包含了exif的样本路径导出到txt. Defaults to None.

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

    have_exif =[]
    for sample in tqdm(
                dataset,
                desc="exif 检查进度:",
                dynamic_ncols=True,
                colour="green",
            ):
        img=Image.open(sample["filepath"])
        if "exif" in img.info:
            have_exif.append(sample["filepath"])
            if clean_inplace:
                piexif.remove(sample["filepath"])

    if log_path:
        with open(log_path,"w") as fw:
            fw.writelines([x+"\n" for x in have_exif])

    print("Here is exif image path:")
    pprint(have_exif)

    return imgslist2dataview(have_exif,dataset)


