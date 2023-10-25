# -*- coding: utf-8 -*-
"""
@Author: captainfffsama
@Date: 2023-02-28 16:52:46
@LastEditors: captainsama tuanzhangsama@outlook.com
@LastEditTime: 2023-03-02 10:00:03
@FilePath: /dataset_manager/core/tools/exporter.py
@Description:
"""
from typing import Optional,List

import os
from concurrent import futures

import fiftyone as fo
import fiftyone.core.dataset as focd
from fiftyone.utils.coco import COCODetectionDatasetExporter
from fiftyone.utils.yolo import YOLOv4DatasetExporter,YOLOv5DatasetExporter

from core.utils import _export_one_sample_anno,_export_one_sample
from core.exporter import SGCCGameDatasetExporter
from core.logging import logging

from core.cache import WEAK_CACHE
from .common_tools import print_time_deco


@print_time_deco
def export_anno_file(
    save_dir: str,
    dataset: Optional[focd.Dataset] = None,
    backup_dir: Optional[str] = None,
    export_classes: Optional[List[str]] = None
):
    """导出数据集的anno文件到 save_dir

    Args:
        save_dir (str): 保存anno的目录
        dataset (focd.Dataset,optional): 需要导出的数据集,若没有就用全局的数据集
        backup_dir (str,optional) = None: 设置anno备份的目录,若不设置为None就不备份
        export_classes (list,optional) = None: 导出的类别,默认为None将导出所有类别
    """
    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is None:
            logging.warning("no dataset in cache,no thing export")
            return
        else:
            dataset = s.dataset
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if backup_dir is not None:
        if not os.path.exists(backup_dir):
            os.mkdir(backup_dir)
    with futures.ThreadPoolExecutor(48) as exec:
        tasks = (
            exec.submit(_export_one_sample_anno, sample, save_dir, backup_dir,export_classes)
            for sample in dataset
        )

        with fo.ProgressBar(total=len(dataset),
                            start_msg="anno导出进度:",
                            complete_msg="anno导出完毕") as pb:
            for task in pb(futures.as_completed(tasks)):
                save_path = task.result()



FORMAT_CLASS_MAP={
    "voc": SGCCGameDatasetExporter,
    "coco": COCODetectionDatasetExporter,
    "yolov4": YOLOv4DatasetExporter,
    "yolov5": YOLOv5DatasetExporter,
}

@print_time_deco
def export_sample(save_dir: str,
                  dataset: Optional[focd.Dataset] = None,
                  get_anno=True,
                  format:str="voc",
                  export_class:Optional[List[str]]=None,
                  label_field:str="ground_truth",
                  **kwargs):
    """导出样本的媒体文件,标签文件和anno文件

    Args:
        save_dir (str): 导出文件的目录
        dataset (focd.Dataset,optional): 需要导出的数据集,若没有就用全局的数据集
        get_anno (bool, optional): 是否导出anno. Defaults to True.
        format (str,optional): 导出格式,必须是 "voc","coco","yolov4","yolov5"之一,默认voc
        export_class (List[str],optinal): 默认为None,导出所有标签.若传入列表等可迭代对象,就仅导出指定的标签
        label_field (str): 保存标签的字段
        **kwargs: 支持``SGCCGameDatasetExporter`` 的参数
    """
    if format not in FORMAT_CLASS_MAP.keys():
        raise ValueError("format must be in {}".format(str(FORMAT_CLASS_MAP.keys())))
    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is None:
            logging.warning("no dataset in cache,no thing export")
            return
        else:
            dataset = s.dataset
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if "export_dir" in kwargs:
        kwargs.pop("export_dir")

    exporter =FORMAT_CLASS_MAP[format](export_dir=save_dir,**kwargs)
    with exporter:
        exporter.log_collection(dataset)
        with futures.ThreadPoolExecutor(48) as exec:
            tasks = (
                exec.submit(_export_one_sample, sample, exporter, get_anno,
                            save_dir,export_class,label_field) for sample in dataset
            )
            with fo.ProgressBar(total=len(dataset),
                                start_msg="样本导出进度:",
                                complete_msg="样本导出完毕") as pb:
                for task in pb(futures.as_completed(tasks)):
                    result = task.result()
