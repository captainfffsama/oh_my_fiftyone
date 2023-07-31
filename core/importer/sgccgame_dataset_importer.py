# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-02-23 09:48:44
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-07-31 19:12:01
@FilePath: /dataset_manager/core/importer/sgccgame_dataset_importer.py
@Description:
'''

import os
from typing import Optional, Tuple, List
import json
import base64

import numpy as np

import fiftyone as fo
import fiftyone.utils.data as foud
import fiftyone.core.metadata as fom
import fiftyone.core.labels as fol

from core.utils import get_all_file_path, parse_xml_info, parse_img_metadata, normalization_xyxy, md5sum
from core.logging import logging


# logging.disable()
class SGCCGameDatasetImporter(foud.LabeledImageDatasetImporter):
    """使用官方API读入数据,但是似乎有性能瓶颈,不推荐使用
    """

    def __init__(self,
                 dataset_dir: Optional[str] = None,
                 img_ext: tuple = ('.jpg', '.JPG', '.bmp', '.BMP', '.jpeg',
                                   '.JPEG', '.png', '.PNG'),
                 shuffle: bool = False,
                 seed=None,
                 max_samples=None):
        """
        Args:
            dataset_dir: the dataset directory
            shuffle (False): whether to randomly shuffle the order in which the
                samples are imported
            seed (None): a random seed to use when shuffling
            max_samples (None): a maximum number of samples to import. By default,
                all samples are imported
        """
        super().__init__(dataset_dir=dataset_dir,
                         shuffle=shuffle,
                         seed=seed,
                         max_samples=max_samples)

        self._img_exts = img_ext
        self._imgs_path = []
        self._dataset_dir = dataset_dir

    def __len__(self):
        return len(self._imgs_path)

    def __iter__(self):
        self._imgs_path_iter = iter(self._imgs_path)
        return self

    def __next__(self):
        current_img_path = next(self._imgs_path_iter)

        if not os.path.exists(current_img_path):
            logging.warning("{} is not exists".format(current_img_path))
            return current_img_path, fom.ImageMetadata(), None

        current_anno_path = os.path.splitext(current_img_path)[0] + ".anno"
        img_meta = parse_img_metadata(current_img_path)

        current_xml_path = os.path.splitext(current_img_path)[0] + ".xml"
        if not os.path.exists(current_xml_path):
            logging.warning("{} do not have xml!".format(current_img_path))
            return current_img_path, img_meta, None

        _, objs_info = parse_xml_info(current_xml_path)
        label_info = objs_infomap2foDetections(objs_info, img_meta)

        return current_img_path, img_meta, {
            "ground_truth": label_info,
        }

    @property
    def has_dataset_info(self):
        return False

    @property
    def has_image_metadata(self):
        """Whether this importer produces
        :class:`fiftyone.core.metadata.ImageMetadata` instances for each image.
        """
        return True

    @property
    def label_cls(self):
        return {
            "ground_truth": fol.Detections,
        }

    def setup(self):
        self._imgs_path = get_all_file_path(self._dataset_dir,
                                            filter_=(".jpg", ".JPG", ".png",
                                                     ".PNG", ".bmp", ".BMP",
                                                     ".jpeg", ".JPEG"))
        self._imgs_path.sort()


def objs_infomap2foDetections(
        objs_info: dict, img_metadata: fom.ImageMetadata) -> fol.Detections:
    result = []
    for k, v in objs_info.items():
        for obj in v:
            bbox, flag = normalization_xyxy(obj, img_metadata.width,
                                            img_metadata.height)
            if not flag:
                result.append(
                    fol.Detection(label=k,
                                  bounding_box=bbox,
                                  ori_data_prob=True))
            else:
                result.append(
                    fol.Detection(label=k,
                                  bounding_box=bbox,
                                  ori_data_prob=False))
    return fol.Detections(detections=result)


def parse_sample_info(
        img_path) -> Tuple[Optional[fo.ImageMetadata], fol.Detections, dict]:
    anno_dict = {}
    if not os.path.exists(img_path):
        logging.warning("{} is not exists".format(img_path))
        return None, fol.Detections(), anno_dict
    img_meta = parse_img_metadata(img_path)
    anno_path = os.path.splitext(img_path)[0] + ".anno"

    xml_path = os.path.splitext(img_path)[0] + ".xml"
    if not os.path.exists(xml_path):
        logging.warning("{} do not have xml!".format(img_path))
        return img_meta, fol.Detections(), anno_dict

    _, objs_info = parse_xml_info(xml_path)
    label_info = objs_infomap2foDetections(objs_info, img_meta)

    if not os.path.exists(anno_path):
        logging.debug("{} do not have anno!".format(img_path))
        anno_dict["chiebot_ID"] = "game_" + md5sum(
            img_path) if not os.path.basename(img_path).startswith(
                "game_") else os.path.splitext(os.path.basename(img_path))[0]
        return img_meta, label_info, anno_dict

    with open(anno_path, 'r') as fr:
        anno = json.load(fr)
    anno_dict["chiebot_ID"] = anno.get("ID", "game_" + md5sum(img_path))
    data_source = anno.get("data_source","Unknow")
    data_source = [data_source] if isinstance(data_source,
                                              str) else data_source
    anno_dict["data_source"] = data_source
    anno_dict["img_quality"] = int(anno.get("img_quality", 0))
    anno_dict["additions"] = anno.get("additions", {})
    anno_dict["tags"] = anno.get("sample_tags", [])
    anno_dict["chiebot_sample_tags"]=anno.get("chiebot_sample_tags",[])

    if anno.get("embedding",None) is not None:
        anno_dict["embedding"] =np.frombuffer(base64.b64decode(anno["embedding"].encode("utf-8")))

    return img_meta, label_info, anno_dict


def generate_sgcc_sample(img_path,
                         extra_attr: Optional[dict] = None
                         ) -> Optional[fo.Sample]:

    img_meta, label_info, anno_dict = parse_sample_info(img_path)
    if img_meta is None:
        return None
    sample = fo.Sample(filepath=img_path, metadata=img_meta)
    sample.add_labels(dict(ground_truth=label_info))
    for k, v in anno_dict.items():
        sample[k] = v

    if extra_attr:
        for k, v in extra_attr.items():
            sample[k] = v

    xml_path = os.path.splitext(img_path)[0] + ".xml"
    if os.path.exists(xml_path):
        sample["xml_md5"] = md5sum(xml_path)

    return sample
