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
from prompt_toolkit.patch_stdout import patch_stdout

from prompt_toolkit.shortcuts import ProgressBar
from tqdm import tqdm

import fiftyone as fo
from core.importer import SGCCGameDatasetImporter, generate_sgcc_sample
from core.utils import get_all_file_path, timeblock
from core.logging import logging, logging_path

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
    if import_error:
        logging.critical("===================================")
        for i in import_error:
            logging.critical(i)
        print("same error happened in import,please check {}".format(logging_path))
    return dataset
