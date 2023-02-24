# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-02-24 10:28:41
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-02-24 15:22:11
@FilePath: /dataset_manager/core/dataset_generator.py
@Description:
'''
from concurrent import futures

from prompt_toolkit.shortcuts import ProgressBar

import fiftyone as fo
from core.importer import SGCCGameDatasetImporter,generate_sgcc_sample
from core.utils import get_all_file_path,timeblock

SAMPLE_MAX_CACHE=30000

def generate_dataset(data_dir,name=None,use_importer=False):
    if use_importer:
        importer= SGCCGameDatasetImporter(dataset_dir=data_dir)
        dataset=fo.Dataset.from_importer(importer,name=name)
    else:
        imgs_path = get_all_file_path(data_dir,
                                      filter_=(".jpg", ".JPG", ".png",
                                               ".PNG", ".bmp", ".BMP",
                                               ".jpeg", ".JPEG"))
        imgs_path.sort()
        sample_cache=[]
        dataset=fo.Dataset(name=name)
        with ProgressBar(title="数据集解析进度:") as pbar:
            with futures.ThreadPoolExecutor(48) as exec:
                tasks=[exec.submit(generate_sgcc_sample,img_path) for img_path in imgs_path]
                for idx,task in pbar(enumerate(futures.as_completed(tasks)),total=len(imgs_path)):
                    sample=task.result()
                    if sample is None:
                        continue
                    sample_cache.append(sample)
                    if idx!=0 and (not idx%SAMPLE_MAX_CACHE):
                        dataset.add_samples(sample_cache)
                        sample_cache.clear()
                dataset.add_samples(sample_cache,dynamic=True)
                sample_cache.clear()


    return dataset
