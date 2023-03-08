# -*- coding: utf-8 -*-
"""
@Author: captainsama
@Date: 2023-03-08 15:09:19
@LastEditors: captainsama tuanzhangsama@outlook.com
@LastEditTime: 2023-03-08 16:31:52
@FilePath: /dataset_manager/core/tools/dataset_opt.py
@Description:
"""

from typing import Optional, Union, List
from pprint import pprint
from datetime import datetime

import os
import json
from concurrent import futures

import fiftyone as fo
import fiftyone.core.dataset as focd
from tqdm import tqdm

from core.utils import get_sample_field, md5sum, get_all_file_path
from core.exporter.sgccgame_dataset_exporter import SGCCGameDatasetExporter
from core.logging import logging

from core.cache import WEAK_CACHE
from core.importer import parse_sample_info, generate_sgcc_sample
from core.tools import imgslist2dataview


def update_dataset(
    dataset: Optional[focd.Dataset] = None,
    update_imgs_asbase: bool = True,
    sample_path_list: Optional[List[str]] = None,
):
    """更新数据集

    Args:
        dataset (Optional[focd.Dataset], optional):
            若dataset参数为None,那么将使用缓存引用中的dataset,这个dataset通常是全局的dataset

        update_imgs_asbase (bool) =True:
            若为False,更新根据数据集来,遍历数据集,更新其中xml发生变化的样本,该情况下,新的数据
            将不会被添加到数据集中.
            若为True,更新将根据样本文件来,样本文件由参数 ``sample_path_list``来确定,
            遍历样本文件,其中和数据集中记录不一样的或者没有的将被更新进数据集
            若 ``sample_path_list`` 没指定,那么样本文件列表为数据集所在文件夹的样本文件.

        sample_path_list (Optional[List[str]]):
            若为None,将从待更新的数据集所在文件夹的样本文件开始遍历,否则将根据提供的样本文件列表开始遍历
    """

    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            dataset = s.dataset
    update_img_path_list = []
    if update_imgs_asbase:
        if sample_path_list:
            imgs_path = sample_path_list
        else:
            dataset_dir = dataset.info.get(
                "dataset_dir", os.path.split(dataset.first().filepath)[0]
            )
            imgs_path = get_all_file_path(
                dataset_dir,
                filter_=(
                    ".jpg",
                    ".JPG",
                    ".png",
                    ".PNG",
                    ".bmp",
                    ".BMP",
                    ".jpeg",
                    ".JPEG",
                ),
            )
        with dataset.save_context() as context:
            for img_path in tqdm(
                imgs_path,
                desc="数据集更新进度:",
                dynamic_ncols=True,
                colour="green",
            ):
                if img_path in dataset:
                    sample = dataset[img_path]
                    xml_path = os.path.splitext(sample.filepath)[0] + ".xml"
                    if not os.path.exists(xml_path):
                        sample.clear_field("ground_truth")
                        continue
                    xml_md5 = md5sum(xml_path)
                    if sample.has_field("xml_md5"):
                        if sample.get_field("xml_md5") != xml_md5:
                            img_meta, label_info, anno_dict = parse_sample_info(
                                sample.filepath
                            )
                            sample.update_fields(anno_dict)
                            sample.update_fields(
                                {
                                    "metadata": img_meta,
                                    "ground_truth": label_info,
                                    "xml_md5": xml_md5,
                                }
                            )
                            update_img_path_list.append(sample.filepath)
                        context.save(sample)
                else:
                    dataset.add_sample(generate_sgcc_sample(img_path))
        dataset.save()
    else:
        for sample in dataset.iter_samples(
            progress=True, autosave=True, batch_size=0.2
        ):
            xml_path = os.path.splitext(sample.filepath)[0] + ".xml"
            if not os.path.exists(xml_path):
                sample.clear_field("ground_truth")
                continue
            xml_md5 = md5sum(xml_path)
            if sample.has_field("xml_md5"):
                if sample.get_field("xml_md5") != xml_md5:
                    img_meta, label_info, anno_dict = parse_sample_info(sample.filepath)
                    sample.update_fields(anno_dict)
                    sample.update_fields(
                        {
                            "metadata": img_meta,
                            "ground_truth": label_info,
                            "xml_md5": xml_md5,
                        }
                    )

                    update_img_path_list.append(sample.filepath)

    update_dataview = imgslist2dataview(update_img_path_list, dataset)
    update_dataview.tag_samples(str(datetime.now()) + "update")

    session = WEAK_CACHE.get("session", None)
    if session is not None:
        session.refresh()


def add_dataset_fields_by_txt(
    txt_path: Union[str, List[str]],
    fields_dict: Union[str, dict],
    dataset: Optional[focd.Dataset] = None,
):
    """通过txt给特定数据集添加字段,txt中不在数据集的数据将被跳过

    Args:
        txt_path (Union[str,List[str]]): 记录的图片路径的txt,或者一个列表
        fields_dict (Union[str, dict]): 可以是一个json或者一个dict
        dataset (Optional[focd.Dataset], optional): 和其他函数一样,默认是全局数据集. Defaults to None.
    """
    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            dataset = s.dataset

    if isinstance(txt_path, str):
        imgs_path = get_all_file_path(
            txt_path,
            filter_=(".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP", ".jpeg", ".JPEG"),
        )
    else:
        imgs_path = txt_path

    if isinstance(fields_dict, str):
        with open(fields_dict, "r") as fr:
            fields_dict = json.load(fr)

    for sample in tqdm(
        dataset.select_by("filepath", imgs_path).iter_samples(autosave=True),
        total=len(imgs_path),
        desc="字段更新进度:",
        dynamic_ncols=True,
        colour="green",
    ):
        for k, v in fields_dict.items():
            sample.set_field(k, v)

    session = WEAK_CACHE.get("session", None)
    if session is not None:
        session.refresh()


def clean_dataset(
    dataset: Optional[focd.Dataset] = None,
):
    """清除数据库中实际文件已经不存在的样本

    Args:
        dataset (Optional[focd.Dataset], optional): 和其他函数一样,默认是全局数据集. Defaults to None.
    """
    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            dataset = s.dataset

    need_del = []

    for sample in tqdm(
        dataset,
        total=len(dataset),
        desc="数据集检查进度:",
        dynamic_ncols=True,
        colour="green",
    ):
        if not os.path.exists(sample["filepath"]):
            need_del.append(sample["id"])

    dataset.delete_samples(need_del)
    dataset.save()

    session = WEAK_CACHE.get("session", None)
    if session is not None:
        session.refresh()
