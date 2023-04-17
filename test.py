# -*- coding: utf-8 -*-
'''
@Author: captainsama
@Date: 2023-03-08 10:13:39
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-04-17 15:38:27
@FilePath: /dataset_manager/test.py
@Description:
'''
import os
from concurrent import futures
from tqdm import tqdm
import numpy as np

from PIL import Image
import piexif
from core.utils import NMS
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
    a=np.array([[0,0,4,4,0.5],[1,1,4,4,0.3],[2,2,6,6,0.7]])
    print(NMS(a))