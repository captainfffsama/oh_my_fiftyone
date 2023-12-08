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
from datetime import datetime
import traceback


from prompt_toolkit.shortcuts import ProgressBar
from tqdm import tqdm

import fiftyone as fo
import fiftyone.core.labels as fol
from core.importer import SGCCGameDatasetImporter, generate_sgcc_sample
from core.exporter import SGCCGameDatasetExporter
from core.utils import get_all_file_path, timeblock,fol_det_nms,_export_one_sample,return_now_time
from core.logging import logging, logging_path
from core.tools import update_dataset,add_dataset_fields_by_txt,imgslist2dataview
from core.cfg import BAK_DIR,SAMPLE_MAX_CACHE


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

        with futures.ThreadPoolExecutor() as exec:
            tasks = (
                exec.submit(generate_sgcc_sample, img_path, extra_attr)
                for img_path in imgs_path
            )
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
                    import_error.append((e,traceback.format_exc(limit=-1)))
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
        logging.critical("=================CRIRICAL==================")
        for i in import_error:
            logging.critical(i[-1])
        print("same error happened in import,please check {}".format(logging_path))
    return dataset

def _deal_sample(img_path,dst_dir,flag,dataset:fo.Dataset,exporter,iou_thr,import_data_cls,back_dir):
    if not os.path.exists(os.path.join(dst_dir,os.path.basename(img_path))):
        return _copy_sample(img_path,dst_dir)
    else:

        ori_sample_path=os.path.join(dst_dir,os.path.basename(img_path))

        ori_xml_path=os.path.splitext(ori_sample_path)[0]+".xml"
        ori_anno_path=os.path.splitext(ori_sample_path)[0]+".anno"
        if os.path.exists(ori_xml_path):
            shutil.copy(ori_xml_path,back_dir)
        if os.path.exists(ori_anno_path):
            shutil.copy(ori_anno_path,back_dir)

    if "overlap" == flag:
        need_import_sample=generate_sgcc_sample(img_path)
        exist_sample=dataset[os.path.join(dst_dir,os.path.basename(img_path))]

        ni_s_label= need_import_sample.ground_truth.detections if need_import_sample.has_field("ground_truth") else []
        e_s_label=exist_sample.ground_truth.detections if exist_sample.has_field("ground_truth") else []

        if len(import_data_cls):
            ni_all_classes=set(import_data_cls)
        else:
            ni_all_classes=set([x.label for x in ni_s_label])

        final_label=[]
        for idx,i in enumerate(ni_s_label):
            if i.label in ni_all_classes:
                final_label.append(i)

        for idx,i in enumerate(e_s_label):
            if i.label not in ni_all_classes:
                final_label.append(i)

        exist_sample.ground_truth=fol.Detections(detections=final_label)
        _export_one_sample(exist_sample,exporter,True,os.path.dirname(exist_sample.filepath))

        return os.path.join(dst_dir,os.path.basename(img_path))
    elif "merge" == flag:
        need_import_sample=generate_sgcc_sample(img_path)
        exist_sample=dataset[os.path.join(dst_dir,os.path.basename(img_path))]

        ni_s_label= need_import_sample.ground_truth.detections if need_import_sample.has_field("ground_truth") else []
        e_s_label=exist_sample.ground_truth.detections if exist_sample.has_field("ground_truth") else []

        for idx,i in enumerate(ni_s_label):
            ni_s_label[idx].confidence=1.0

        for idx,i in enumerate(e_s_label):
            e_s_label[idx].confidence=0.7

        e_s_label.extend(ni_s_label)

        final_label=fol_det_nms(e_s_label,iou_thr=iou_thr,sort_by="score")
        exist_sample.ground_truth=final_label
        _export_one_sample(exist_sample,exporter,True,os.path.dirname(exist_sample.filepath))

        return os.path.join(dst_dir,os.path.basename(img_path))
    elif "new" == flag:
        need_import_sample=generate_sgcc_sample(img_path)
        exist_sample=dataset[os.path.join(dst_dir,os.path.basename(img_path))]
        exist_sample.ground_truth=need_import_sample.ground_truth
        _export_one_sample(exist_sample,exporter,True,os.path.dirname(exist_sample.filepath))

        return os.path.join(dst_dir,os.path.basename(img_path))


def _copy_sample(img_path,dst_dir) -> str:
    xml_path=os.path.splitext(img_path)[0]+".xml"
    anno_path=os.path.splitext(img_path)[0]+".anno"
    shutil.copy(img_path,dst_dir)
    if os.path.exists(xml_path):
        shutil.copy(xml_path,dst_dir)

    if os.path.exists(anno_path):
        shutil.copy(anno_path,dst_dir)

    return os.path.join(dst_dir,os.path.basename(img_path))

def import_new_sample2exist_dataset(exist_dataset:fo.Dataset,new_samples_path:str,same_sample_deal:str,merge_iou_thr=0.7,import_data_cls=()):
    extra_attr_cfg_file = get_all_file_path(new_samples_path, filter_=(".annocfg",))

    extra_attr = {}
    if extra_attr_cfg_file:
        if os.path.exists(extra_attr_cfg_file[0]):
            try:
                with open(extra_attr_cfg_file[0], "r") as fr:
                    extra_attr = json.load(fr)
            except json.JSONDecodeError as e:
                print("{} 不是json标准格式,报错信息如下:{}".format(extra_attr_cfg_file[0],e.msg))
                return

    imgs_path = get_all_file_path(
            new_samples_path,
            filter_=(".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP", ".jpeg", ".JPEG"),
        )

    # TODO: 这里需要注意以下,默认的路径
    dst_dir=exist_dataset.info.get("dataset_dir",os.path.split(exist_dataset.first().filepath)[0])

    new_imgs_path=[]


    back_dir=os.path.join(BAK_DIR,return_now_time())
    if not os.path.exists(back_dir):
        os.makedirs(back_dir)
    exporter = SGCCGameDatasetExporter(export_dir=dst_dir)
    import_error = []

    with exporter:
        with futures.ThreadPoolExecutor() as exec:
            tasks=(exec.submit(_deal_sample,img_path,dst_dir,same_sample_deal,exist_dataset,exporter,merge_iou_thr,import_data_cls,back_dir) for img_path in imgs_path)
            for task in tqdm(futures.as_completed(tasks),
                    total=len(imgs_path),
                    desc="样本拷贝合并进度:",
                    dynamic_ncols=True,
                    colour="green",
                ):

                try:
                    new_img_path=task.result()
                except Exception as e:
                    new_img_path=None
                    import_error.append((e,traceback.format_exc(limit=-1)))
                    continue
                if new_img_path:
                    new_imgs_path.append(new_img_path)

    update_dataset(exist_dataset,update_imgs_asbase=True,sample_path_list=new_imgs_path)


    if extra_attr:
        add_dataset_fields_by_txt(new_imgs_path,extra_attr,exist_dataset)

    imgslist2dataview(new_imgs_path,exist_dataset).tag_samples(str(datetime.now().replace(microsecond=0))+"import")

    if import_error:
        logging.critical("=================CRIRICAL==================")
        for i in import_error:
            logging.critical(i[-1])
        print("same error happened in import,please check {}".format(logging_path))