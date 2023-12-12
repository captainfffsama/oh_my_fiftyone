# -*- coding: utf-8 -*-
"""
@Author: captainfffsama
@Date: 2023-12-08 16:19:03
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-12-08 16:47:11
@FilePath: /oh_my_fiftyone/core/tools/xml_change/xml_tools.py
@Description:
"""
from datetime import datetime
import shutil
import os
from typing import Union, Tuple, Optional, Dict
from concurrent import futures

import fiftyone as fo
from fiftyone.core.dataset import Dataset
from fiftyone.core.view import DatasetView
import fiftyone.core.labels as fol
from fiftyone.utils import voc

from prompt_toolkit import prompt
from prompt_toolkit.validation import Validator
from prompt_toolkit.completion import WordCompleter

from core.cfg import BAK_DIR
from core.exporter.sgccgame_dataset_exporter import SGCCGameDatasetExporter
from core.cache import WEAK_CACHE
from ..dataset_opt import update_dataset


def _change_xml_cls_deal_one(
    img_path: str,
    need_del_classes: Tuple[str],
    change_cls_map: Dict[str, str],
    bak_dir: str,
) -> Tuple[bool,str]:
    xml_path = os.path.splitext(img_path)[0] + ".xml"
    if not os.path.exists(xml_path):
        return False,img_path

    objs_info = voc.load_voc_detection_annotations(xml_path).to_detections().detections
    remain_objs = []
    change_flag = False
    for obj in objs_info:
        if obj.label in need_del_classes:
            change_flag = True
            continue
        if obj.label in change_cls_map:
            change_flag = True
            obj.label = change_cls_map[obj.label]
        remain_objs.append(obj)

    if not change_flag:
        return False,img_path

    save_path = os.path.join(bak_dir, os.path.basename(xml_path))
    shutil.copyfile(xml_path, save_path)

    saver = SGCCGameDatasetExporter(
        export_dir=os.path.split(xml_path)[0], export_media=False
    )
    with saver:
        saver.export_sample(img_path, detections=fol.Detections(detections=remain_objs))
    return True,img_path


def change_xml_label_class(
    dataset: Union[Dataset, DatasetView],
    need_del_classes: Optional[Tuple[str]] = (),
    change_cls_map: Optional[Dict[str, str]] = None,
    bak_dir: Optional[str] = None,
):
    """更改dataset中的xml标签.可以删除标签或者修改标签.两个参数都指定的时候,对于一个样本,是先删除不要的类别,
        然后再修改标签

    Args:
        dataset (Union[Dataset, DatasetView]): 需要处理xml的数据集
        need_del_classes (Optional[Tuple[str]], optional): 需要删除的标签的名称. Defaults to ().
        change_cls_map (Optional[Dict[str, str]], optional): 需要修改的标签的映射,key为需要修改的标签,value为修改后的标签. Defaults to None.
        bak_dir (Optional[str], optional): 原始xml备份文件夹. 没指定就是/tmp/oh_my_fiftyone/bak/当前时间original_xml.
    """
    if change_cls_map is None:
        change_cls_map = {}

    valida = Validator.from_callable(
        lambda x: x in ("y", "n"), error_message="瞎选什么啊"
    )

    if bak_dir is None:
        now_time = datetime.now().replace(microsecond=0)
        now_time = str(now_time).replace(" ", "_").replace(":", "_")
        bak_dir = os.path.join(BAK_DIR, now_time + "original_xml")
    if not os.path.exists(bak_dir):
        os.makedirs(bak_dir)

    if isinstance(dataset, Dataset):
        deal_dataset = dataset
    else:
        deal_dataset = fo.load_dataset(dataset.dataset_name)

    flag = prompt(
        "你将准备修改xml的类别,需要删除的类别为{} \n,需要修改名称的类别为{} \n,备份文件夹为{}\n,是否继续?".format(
            need_del_classes, change_cls_map, bak_dir
        ),
        validator=valida,
        completer=WordCompleter(["y", "n"]),
    )
    if "n" == flag:
        print("取消修改")
        return

    need_deal_img_paths = dataset.values("filepath")
    have_change_list=[]

    with fo.ProgressBar(
        total=len(need_deal_img_paths),
        start_msg="标签文件修改进度:",
        complete_msg="修改完毕",
    ) as pb:
        with futures.ThreadPoolExecutor(max_workers=5) as executor:
            tasks = [
                executor.submit(
                    _change_xml_cls_deal_one,
                    img_path,
                    need_del_classes,
                    change_cls_map,
                    bak_dir,
                )
                for img_path in need_deal_img_paths
            ]
            for task in futures.as_completed(tasks):
                change_flag,img_path = task.result()
                if change_flag:
                    have_change_list.append(img_path)
                pb.update(1)

    update_func = update_dataset.__wrapped__
    update_func (
        dataset=deal_dataset,
        update_imgs_asbase=True,
        sample_path_list=have_change_list,
    )

    s: fo.Session = WEAK_CACHE.get("session", None)
    if s is not None:
        s.refresh()