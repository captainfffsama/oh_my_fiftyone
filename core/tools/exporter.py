# -*- coding: utf-8 -*-
"""
@Author: captainfffsama
@Date: 2023-02-28 16:52:46
@LastEditors: captainsama tuanzhangsama@outlook.com
@LastEditTime: 2023-03-02 10:00:03
@FilePath: /dataset_manager/core/tools/exporter.py
@Description:
"""
from typing import Optional, Union, List

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


def _export_one_sample_anno(sample, save_dir):
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

    result["chiebot_sample_tags"] = get_sample_field(sample,
                                                     "chiebot_sample_tags",
                                                     default=[])

    result["img_shape"] = (
        sample["metadata"].height,
        sample["metadata"].width,
        sample["metadata"].num_channels,
    )
    result["objs_info"] = []
    dets = get_sample_field(sample, "ground_truth")
    if dets:
        for det in dets.detections:
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

    save_path = os.path.join(save_dir,
                             os.path.splitext(sample.filename)[0] + ".anno")
    with open(save_path, "w") as fw:
        json.dump(result, fw, indent=4, sort_keys=True)


def export_anno_file(
    save_dir: str,
    dataset: Optional[focd.Dataset] = None,
):
    """导出数据集的anno文件到 save_dir

    Args:
        save_dir (str): 保存anno的目录
        dataset (focd.Dataset,optional): 需要导出的数据集,若没有就用全局的数据集
    """
    if dataset is None:
        s=WEAK_CACHE.get("session",None)
        if s is None:
            logging.warning("no dataset in cache,no thing export")
            return
        else:
            dataset=s.dataset
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with futures.ThreadPoolExecutor(48) as exec:
        tasks = [
            exec.submit(_export_one_sample_anno, sample, save_dir)
            for sample in dataset
        ]
        for task in tqdm(
                futures.as_completed(tasks),
                total=len(dataset),
                desc="anno导出进度:",
                dynamic_ncols=True,
                colour="green",
        ):
            result = task.result()


def _export_one_sample(sample, exporter, get_anno, save_dir):
    image_path = sample.filepath

    metadata = sample.metadata
    if exporter.requires_image_metadata and metadata is None:
        metadata = fo.ImageMetadata.build_for(image_path)

    # Assumes single label field case
    label = sample["ground_truth"]

    exporter.export_sample(image_path, label, metadata=metadata)

    if get_anno:
        _export_one_sample_anno(sample, save_dir)


def export_sample(save_dir: str,
                  dataset: Optional[focd.Dataset] = None,
                  get_anno=True,
                  **kwargs):
    """导出样本的媒体文件,标签文件和anno文件

    Args:
        save_dir (str): 导出文件的目录
        dataset (focd.Dataset,optional): 需要导出的数据集,若没有就用全局的数据集
        get_anno (bool, optional): 是否导出anno. Defaults to True.
        **kwargs: 支持``SGCCGameDatasetExporter`` 的参数
    """
    if dataset is None:
        s=WEAK_CACHE.get("session",None)
        if s is None:
            logging.warning("no dataset in cache,no thing export")
            return
        else:
            dataset=s.dataset
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if "export_dir" in kwargs:
        kwargs.pop("export_dir")
    exporter = SGCCGameDatasetExporter(export_dir=save_dir, **kwargs)
    with exporter:
        exporter.log_collection(dataset)
        with futures.ThreadPoolExecutor(48) as exec:
            tasks = [
                exec.submit(_export_one_sample, sample, exporter, get_anno,
                            save_dir) for sample in dataset
            ]
            for task in tqdm(
                    futures.as_completed(tasks),
                    total=len(dataset),
                    desc="样本导出进度:",
                    dynamic_ncols=True,
                    colour="green",
            ):
                result = task.result()


def update_dataset(dataset: Optional[focd.Dataset] = None,
                   update_imgs_asbase: bool = True,
                   sample_path_list: Optional[List[str]] = None):
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
        s=WEAK_CACHE.get("session",None)
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            dataset=s.dataset

    if update_imgs_asbase:
        if sample_path_list:
            imgs_path = sample_path_list
        else:
            dataset_dir = dataset.info.get(
                "dataset_dir",
                os.path.split(dataset.first().filepath)[0])
            imgs_path = get_all_file_path(
                dataset_dir,
                filter_=(".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP",
                         ".jpeg", ".JPEG"),
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
                                sample.filepath)
                            sample.update_fields(anno_dict)
                            sample.update_fields({
                                "metadata": img_meta,
                                "ground_truth": label_info,
                                "xml_md5": xml_md5
                            })
                        context.save(sample)
                else:
                    dataset.add_sample(generate_sgcc_sample(img_path))
        dataset.save()
    else:
        for sample in dataset.iter_samples(progress=True,
                                           autosave=True,
                                           batch_size=0.2):
            xml_path = os.path.splitext(sample.filepath)[0] + ".xml"
            if not os.path.exists(xml_path):
                sample.clear_field("ground_truth")
                continue
            xml_md5 = md5sum(xml_path)
            if sample.has_field("xml_md5"):
                if sample.get_field("xml_md5") != xml_md5:
                    img_meta, label_info, anno_dict = parse_sample_info(
                        sample.filepath)
                    sample.update_fields(anno_dict)
                    sample.update_fields({
                        "metadata": img_meta,
                        "ground_truth": label_info,
                        "xml_md5": xml_md5
                    })
    session = WEAK_CACHE.get("session", None)
    if session is not None:
        session.refresh()


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
        dataset=session.dataset
    if dataset and session:
        if txt_path is not None:
            if os.path.exists(txt_path):
                imgs_path = get_all_file_path(txt_path)
                return dataset.select_by("filepath", imgs_path)
        else:
            return dataset.select(session.selected)
    return None


def add_dataset_fields_by_txt(txt_path: Union[str, List[str]],
                              fields_dict: Union[str, dict],
                              dataset: Optional[focd.Dataset] = None):
    """通过txt给特定数据集添加字段,txt中不在数据集的数据将被跳过

    Args:
        txt_path (Union[str,List[str]]): 记录的图片路径的txt,或者一个列表
        fields_dict (Union[str, dict]): 可以是一个json或者一个dict
        dataset (Optional[focd.Dataset], optional): 和其他函数一样,默认是全局数据集. Defaults to None.
    """
    if dataset is None:
        s=WEAK_CACHE.get("session",None)
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            dataset=s.dataset

    if isinstance(txt_path, str):
        imgs_path = get_all_file_path(
            txt_path,
            filter_=(".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP", ".jpeg",
                     ".JPEG"),
        )
    else:
        imgs_path = txt_path

    if isinstance(fields_dict, str):
        with open(fields_dict, "r") as fr:
            fields_dict = json.load(fr)

    with dataset.save_context() as context:
        for sample in tqdm(
                dataset.select_by("filepath", imgs_path),
                total=len(imgs_path),
                desc="字段更新进度:",
                dynamic_ncols=True,
                colour="green",
        ):
            for k, v in fields_dict.items():
                sample[k] = v
            context.save(sample)

    session = WEAK_CACHE.get("session", None)
    if session is not None:
        session.refresh()
