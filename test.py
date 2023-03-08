# -*- coding: utf-8 -*-
'''
@Author: captainsama
@Date: 2023-03-08 10:13:39
@LastEditors: captainsama tuanzhangsama@outlook.com
@LastEditTime: 2023-03-08 15:29:21
@FilePath: /dataset_manager/test.py
@Description:
'''
import os
from concurrent import futures
from tqdm import tqdm

from PIL import Image
import piexif
def get_all_file_path(file_dir:str,filter_=('.jpg')) -> list:
    #遍历文件夹下所有的file
    return [os.path.join(maindir,filename) for maindir,_,file_name_list in os.walk(file_dir) \
        for filename in file_name_list \
        if os.path.splitext(filename)[1] in filter_ ]


def deal_one(img_path):
    img=Image.open(img_path)
    if "exif" in img.info:
        return img_path
    else:
        return None

def check_exif(img_dir,log_path):
    imgs_path=get_all_file_path(img_dir)

    result=[]
    with futures.ThreadPoolExecutor(48) as exec:
        tasks=[exec.submit(deal_one,img_path) for img_path in imgs_path]
        for task in tqdm(futures.as_completed(tasks),total=len(imgs_path)):
            img_path=task.result()
            if img_path:
                result.append(img_path+"\n")

    with open(log_path,"w") as fw:
        fw.writelines(result)

if __name__ == "__main__":
    img_dir="/home/gpu-server/project/data/game_new/data_tmp"
    log_path="/tmp/have_exif.log"
    check_exif(img_dir,log_path)

