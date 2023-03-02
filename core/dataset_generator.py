# -*- coding: utf-8 -*-
"""
@Author: captainfffsama
@Date: 2023-02-24 10:28:41
@LastEditors: captainsama tuanzhangsama@outlook.com
@LastEditTime: 2023-03-02 09:59:53
@FilePath: /dataset_manager/core/dataset_generator.py
@Description:
"""
import os
import json
from concurrent import futures
import shutil


from prompt_toolkit.shortcuts import ProgressBar
from tqdm import tqdm

import fiftyone as fo
from core.importer import SGCCGameDatasetImporter, generate_sgcc_sample
from core.utils import get_all_file_path, timeblock
from core.logging import logging, logging_path
from core.tools import update_dataset,add_dataset_fields_by_txt

SAMPLE_MAX_CACHE = 60000


def generate_dataset(data_dir, name=None, use_importer=False, persistent=True):
    """不推荐使用 importer导入

    Args:
        data_dir (_type_): _description_
        name (_type_, optional): _description_. Defaults to None.
        use_importer (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    import_error = []
    if use_importer:
        importer = SGCCGameDatasetImporter(dataset_dir=data_dir)
        dataset = fo.Dataset.from_importer(importer, name=name, overwrite=True)
    else:
        imgs_path = get_all_file_path(
            data_dir,
            filter_=(".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP", ".jpeg", ".JPEG"),
        )
        extra_attr_cfg_file = get_all_file_path(data_dir, filter_=(".annocfg",))

        extra_attr = None
        if extra_attr_cfg_file:
            if os.path.exists(extra_attr_cfg_file[0]):
                with open(extra_attr_cfg_file[0], "r") as fr:
                    extra_attr = json.load(fr)

        imgs_path.sort()
        sample_cache = []
        dataset = fo.Dataset(name=name, overwrite=True)

        with futures.ThreadPoolExecutor(48) as exec:
            tasks = [
                exec.submit(generate_sgcc_sample, img_path, extra_attr)
                for img_path in imgs_path
            ]
            for idx, task in tqdm(
                enumerate(futures.as_completed(tasks)),
                total=len(imgs_path),
                desc="数据集解析进度:",
                dynamic_ncols=True,
                colour="green",
            ):
                try:
                    sample = task.result()
                except Exception as e:
                    sample = None
                    import_error.append(e)
                    # raise e
                if sample is None:
                    continue
                sample_cache.append(sample)
                if idx != 0 and (not idx % (SAMPLE_MAX_CACHE - 1)):
                    dataset.add_samples(sample_cache)
                    sample_cache.clear()

            dataset.add_samples(sample_cache)
            sample_cache.clear()
    dataset.persistent = persistent
    if os.path.isdir(data_dir):
        dataset.info["dataset_dir"]=data_dir
        dataset.save()
    if import_error:
        logging.critical("===================================")
        for i in import_error:
            logging.critical(i)
        print("same error happened in import,please check {}".format(logging_path))
    return dataset

def _copy_sample(img_path,dst_dir) -> str:
    xml_path=os.path.splitext(img_path)[0]+".xml"
    anno_path=os.path.splitext(img_path)[0]+".anno"
    shutil.copy(img_path,dst_dir)
    if os.path.exists(xml_path):
        shutil.copy(xml_path,dst_dir)

    if os.path.exists(anno_path):
        shutil.copy(anno_path,dst_dir)

    return os.path.join(dst_dir,os.path.basename(img_path))

def import_new_sample2exist_dataset(exist_dataset:fo.Dataset,new_samples_path:str):
    imgs_path = get_all_file_path(
            new_samples_path,
            filter_=(".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP", ".jpeg", ".JPEG"),
        )

    dst_dir=exist_dataset.info.get("dataset_dir",os.path.split(exist_dataset.first().filepath)[0])

    new_imgs_path=[]
    with futures.ThreadPoolExecutor(32) as exec:
        tasks=[exec.submit(_copy_sample,img_path,dst_dir) for img_path in imgs_path]
        for task in tqdm(futures.as_completed(tasks),
                total=len(imgs_path),
                desc="样本拷贝进度:",
                dynamic_ncols=True,
                colour="green",
            ):
            new_imgs_path.append(task.result())

    update_dataset(exist_dataset,update_imgs_asbase=True,sample_path_list=new_imgs_path)

    extra_attr_cfg_file = get_all_file_path(new_samples_path, filter_=(".annocfg",))

    extra_attr = {}
    if extra_attr_cfg_file:
        if os.path.exists(extra_attr_cfg_file[0]):
            with open(extra_attr_cfg_file[0], "r") as fr:
                extra_attr = json.load(fr)

    if extra_attr:
        add_dataset_fields_by_txt(new_imgs_path,extra_attr,exist_dataset)

