# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-04-14 14:18:20
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-04-18 10:45:09
@FilePath: /dataset_manager/core/parse_label.py
@Description:
'''
from concurrent import futures
import os
import yaml
from typing import Dict,List

import fiftyone as fo
from fiftyone.utils import voc,coco,yolo
import fiftyone.core.labels as fol

from core.utils import get_all_file_path
from core.logging import logging


def guess_label_type(label_dir):
    xmls= get_all_file_path(label_dir,filter_=(".xml"))
    txts= get_all_file_path(label_dir,filter_=(".txt"))
    if xmls or txts:
        if len(xmls)>len(txts):
            return "voc",_get_voc_args(label_dir)
        else:
            if os.path.exists(os.path.join(label_dir,"obj.names")):
                return "yolov4",_get_yolo_args(label_dir)
            elif os.path.exists(os.path.join(label_dir,"dataset.yaml")):
                return "yolov5",_get_yolo_args(label_dir)
            else:
                logging.error("错误:猜不到{}是什么标签,看起来像yolo,但是不规范".format(label_dir))
                print("错误:猜不到{}是什么标签,看起来像yolo,但是不规范".format(label_dir))
                return None,None

    elif os.path.exists(os.path.join(label_dir,"labels.json")):
        return "coco",_get_coco_args(label_dir)
    else:
        logging.error("错误:猜不到{}是什么标签".format(label_dir))
        print("错误:猜不到{}是什么标签".format(label_dir))
        return None,None

def _get_voc_args(label_dir):
    xmls= get_all_file_path(label_dir,filter_=(".xml"))
    if xmls:
        return {"xml_list":xmls}
    else:
        return None

def _get_yolo_args(label_dir):
    txts= get_all_file_path(label_dir,filter_=(".txt"))
    if os.path.exists(os.path.join(label_dir,"obj.names")):
        with open(os.path.join(label_dir,"obj.names"),"r") as fr:
            classes=[x.strip() for x in fr.readlines()]
        return {"txt_list":txts,"classes":classes}
    elif os.path.exists(os.path.join(label_dir,"dataset.yaml")):
        with open(os.path.join(label_dir,"dataset.yaml"), 'r') as fr:
            classes= yaml.load(fr, yaml.FullLoader)["names"]
        return {"txt_list":txts,"classes":classes}
    else:
        logging.error("错误:猜不到{}是什么标签,看起来像yolo,但是不规范".format(label_dir))
        print("错误:猜不到{}是什么标签,看起来像yolo,但是不规范".format(label_dir))
        return None

def _get_coco_args(label_dir):
    if os.path.exists(os.path.join(label_dir,"labels.json")):
        return {"anno_json":os.path.join(label_dir,"labels.json")}
    else:
        logging.error("错误:猜不到{}是什么标签,反正不是coco".format(label_dir))
        print("错误:猜不到{}是什么标签,反正不是coco".format(label_dir))
        return None

def _parse_voc(xml_list,exclude_classes) -> Dict[str,list]:
    deal_one=lambda xml,ex_cls:(os.path.basename(xml).split(".")[0],[x for x in voc.load_voc_detection_annotations(xml).to_detections().detections if x.label not in ex_cls])
    result={}
    with fo.ProgressBar(
            total=len(xml_list), start_msg="标签文件解析进度:", complete_msg="解析完毕"
        ) as pb:
        with futures.ThreadPoolExecutor(48) as exec:
            tasks=[exec.submit(deal_one, xml,exclude_classes) for xml in xml_list]

            for task in pb(futures.as_completed(tasks)):
                name,label=task.result()
                result[name]=label

    return result

def _parse_yolo(txt_list,classes,exclude_classes) -> Dict[str,list]:
    deal_one=lambda txt,classes,ex_cls:(os.path.basename(txt).split(".")[0],[x for x in yolo.load_yolo_annotations(txt,classes).detections if x.label not in ex_cls])
    result={}
    with fo.ProgressBar(
            total=len(txt_list), start_msg="标签文件解析进度:", complete_msg="解析完毕"
        ) as pb:
        with futures.ThreadPoolExecutor(48) as exec:
            tasks=[exec.submit(deal_one,txt,classes,exclude_classes) for txt in txt_list]

            for task in pb(futures.as_completed(tasks)):
                name,label=task.result()
                result[name]=label
    return result


def _parse_coco(anno_json,exclude_classes) -> Dict[str,list]:
    dataset_info,classes,s_map,img_ids_map,anno=coco.load_coco_detection_annotations(anno_json)

    result={}
    with fo.ProgressBar(
            total=len(anno.keys()), start_msg="标签文件解析进度:", complete_msg="解析完毕"
        ) as pb:
        for coco_id,coco_anno_obj in pb(anno.items()):
            img_name=os.path.basename(img_ids_map[coco_id]["file_name"]).split(".")[0]

            img_wh=(img_ids_map[coco_id]["width"],img_ids_map[coco_id]["height"])
            labels=[x.to_detection(img_wh,classes=classes) for x in coco_anno_obj]
            labels_final=[x for x in labels if x.label not in labels]
            result[img_name]=labels
    return result


GET_ARGS_MAP={
    "voc":_get_voc_args,
    "yolov4":_get_yolo_args,
    "yolov5":_get_yolo_args,
    "coco":_get_coco_args
}

PARSER_MAP={
    "voc": _parse_voc,
    "yolov4": _parse_yolo,
    "yolov5": _parse_yolo,
    "coco": _parse_coco
}


def parser_labels(label_dir,exclude_class=None,label_type=None) -> Dict[str,List[fol.Detection]]:
    if label_type is None:
        label_type,label_args=guess_label_type(label_dir)
    else:
        label_args=GET_ARGS_MAP.get(label_type,lambda x:None)(label_dir)
    if label_args is None:
        return
    if exclude_class is None:
        exclude_class=set([])

    return PARSER_MAP[label_type](**label_args,exclude_class=exclude_class)