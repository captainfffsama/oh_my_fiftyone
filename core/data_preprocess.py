# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-03-03 15:40:01
@LastEditors: captainsama tuanzhangsama@outlook.com
@LastEditTime: 2023-03-06 12:47:56
@FilePath: /dataset_manager/core/data_preprocess.py
@Description:
'''
import os
import shutil
from concurrent import futures

import cv2
from tqdm import tqdm
from core.utils import md5sum,get_all_file_path

def preprocess_one(img_path,save_dir,rename=True,convert2jpg=True,rename_prefix="game_"):
    img_path_noe,img_ext=os.path.splitext(img_path)
    xml_path=img_path_noe+".xml"
    if not os.path.exists(xml_path):
        xml_path=xml_path+".XML"

    anno_path=img_path_noe+".anno"

    if rename:
        img_md5=md5sum(img_path)
        new_name=rename_prefix+img_md5
    else:
        new_name=os.path.basename(img_path_noe)

    if convert2jpg:
        img=cv2.imread(img_path,cv2.IMREAD_IGNORE_ORIENTATION|cv2.IMREAD_COLOR)
        if img_ext in (".jpg",".jpeg",".JPG",".JPEG") and len(img.shape)==3 and img.shape[-1]==3:
            shutil.copy(img_path,os.path.join(save_dir,new_name+".jpg"))
        else:
            cv2.imwrite(os.path.join(save_dir,new_name+".jpg"),img)
    else:
        shutil.copy(img_path,os.path.join(save_dir,new_name+".jpg"))

    if os.path.exists(xml_path):
        shutil.copy(xml_path,os.path.join(save_dir,new_name+".xml"))

    if os.path.exists(anno_path):
        shutil.copy(anno_path,os.path.join(save_dir,new_name+".anno"))

    return img_path,os.path.join(save_dir,new_name+".jpg")



def preprocess(datas_dir,save_dir,rename=True,convert2jpg=True,rename_prefix="game_"):
    imgs_path=get_all_file_path(datas_dir)

    result=[]
    with futures.ThreadPoolExecutor(48) as exec:
        tasks=[exec.submit(preprocess_one,img_path,save_dir,rename,convert2jpg,rename_prefix) for img_path in imgs_path]
        for task in tqdm(futures.as_completed(tasks),total=len(imgs_path),desc="预处理进度:"):
            img_path,new_path=task.result()
            result.append(img_path+" => "+new_path+"\n")

    with open(os.path.join(save_dir,"convertlog.log"),"w") as fw:
        fw.writelines(result)

