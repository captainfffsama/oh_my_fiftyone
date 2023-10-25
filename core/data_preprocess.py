# -*- coding: utf-8 -*-
"""
@Author: captainfffsama
@Date: 2023-03-03 15:40:01
@LastEditors: captainsama tuanzhangsama@outlook.com
@LastEditTime: 2023-03-09 12:11:04
@FilePath: /dataset_manager/core/data_preprocess.py
@Description:
"""
from typing import Optional
from copy import deepcopy
import os
import shutil
from concurrent import futures
import json

import cv2
import piexif
from tqdm import tqdm
from core.utils import md5sum, get_all_file_path


def preprocess_one(
    img_path,
    save_dir,
    rename=True,
    convert2jpg=True,
    rename_prefix="game_",
    anno_info: Optional[dict] = None,
):
    img_path_noe, img_ext = os.path.splitext(img_path)
    xml_path = img_path_noe + ".xml"
    if not os.path.exists(xml_path):
        xml_path = xml_path + ".XML"

    anno_path = img_path_noe + ".anno"

    if rename:
        img_md5 = md5sum(img_path)
        new_name = rename_prefix + img_md5
    else:
        new_name = os.path.basename(img_path_noe)

    if convert2jpg:
        img = cv2.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        if img is None:
            print("ERROR!{} is broken".format(img_path))
            return img_path, None
        if (
            img_ext in (".jpg", ".jpeg", ".JPG", ".JPEG")
            and len(img.shape) == 3
            and img.shape[-1] == 3
        ):
            shutil.copy(img_path, os.path.join(save_dir, new_name + ".jpg"))
            try:
                piexif.remove(os.path.join(save_dir, new_name + ".jpg"))
            except Exception as e:
                print("{} remove exif fail,use opencv resave".format(img_path))
                print(e)
                cv2.imwrite(os.path.join(save_dir, new_name + ".jpg"), img)

        else:
            cv2.imwrite(os.path.join(save_dir, new_name + ".jpg"), img)
    else:
        shutil.copy(img_path, os.path.join(save_dir, new_name + ".jpg"))

    if os.path.exists(xml_path):
        shutil.copy(xml_path, os.path.join(save_dir, new_name + ".xml"))

    if os.path.exists(anno_path):
        shutil.copy(anno_path, os.path.join(save_dir, new_name + ".anno"))
    else:
        if anno_info:
            anno_content=deepcopy(anno_info)
            anno_content.update({"addtions": {"ori_jpg_path": img_path}})
            with open(os.path.join(save_dir, new_name + ".anno"), "w") as fw:
                json.dump(anno_content, fw, indent=4)

    return img_path, os.path.join(save_dir, new_name + ".jpg")


def preprocess(
    datas_dir,
    save_dir,
    rename=True,
    convert2jpg=True,
    rename_prefix="game_",
    anno_info=None,
):
    imgs_path = get_all_file_path(datas_dir)

    result = []
    anno_infos = None
    if anno_info:
        try:
            with open(anno_info, "r") as fr:
                anno_infos = json.load(fr)
        except Exception as e:
            print("{} 解析失败,将不会生成anno")

    with futures.ThreadPoolExecutor(48) as exec:
        tasks = (
            exec.submit(
                preprocess_one,
                img_path,
                save_dir,
                rename,
                convert2jpg,
                rename_prefix,
                anno_infos,
            )
            for img_path in imgs_path
        )
        for task in tqdm(
            futures.as_completed(tasks), total=len(imgs_path), desc="预处理进度:"
        ):
            img_path, new_path = task.result()
            if img_path and new_path:
                result.append(img_path + " => " + new_path + "\n")

    with open(os.path.join(save_dir, "convertlog.log"), "w") as fw:
        fw.writelines(result)
