# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-02-23 09:48:44
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-03-01 10:35:52
@FilePath: /dataset_manager/core/importer/sgccgame_dataset_importer.py
@Description:
'''

import os
from typing import Optional, Tuple, List
import json

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

        img_meta["xml_path"] = current_xml_path
        _, objs_info = parse_xml_info(current_xml_path)
        label_info, wrong_obj_flag = objs_infomap2foDetections(
            objs_info, img_meta)

        return current_img_path, img_meta, {
            "ground_truth": label_info,
            "test_filed": {
                "wrong_obj": wrong_obj_flag
            }
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
        return {"ground_truth": fol.Detections, "test_filed": dict}

    def setup(self):
        self._imgs_path = get_all_file_path(self._dataset_dir,
                                            filter_=(".jpg", ".JPG", ".png",
                                                     ".PNG", ".bmp", ".BMP",
                                                     ".jpeg", ".JPEG"))
        self._imgs_path.sort()


def objs_infomap2foDetections(
        objs_info: dict,
        img_metadata: fom.ImageMetadata) -> Tuple[fol.Detections, bool]:
    result = []
    no_wrong_obj = True
    for k, v in objs_info.items():
        for obj in v:
            bbox, flag = normalization_xyxy(obj, img_metadata.width,
                                            img_metadata.height)
            if not flag:
                logging.warning("{} have wrong obj".format(
                    img_metadata.img_path))
                result.append(
                    fol.Detection(label=k,
                                  bounding_box=bbox,
                                  ori_data_prob=True))
                no_wrong_obj = False
            else:
                result.append(
                    fol.Detection(label=k,
                                  bounding_box=bbox,
                                  ori_data_prob=False))
    return fol.Detections(detections=result), no_wrong_obj


def generate_sgcc_sample(img_path,
                         extra_attr: Optional[dict] = None
                         ) -> Optional[fo.Sample]:

    sample = None

    if not os.path.exists(img_path):
        logging.warning("{} is not exists".format(img_path))
        return sample

    img_meta = parse_img_metadata(img_path)
    sample = fo.Sample(filepath=img_path, metadata=img_meta)
    if extra_attr:
        for k, v in extra_attr.items():
            sample[k] = v
    anno_path = os.path.splitext(img_path)[0] + ".anno"

    xml_path = os.path.splitext(img_path)[0] + ".xml"
    if not os.path.exists(xml_path):
        logging.warning("{} do not have xml!".format(img_path))
        return sample

    _, objs_info = parse_xml_info(xml_path)
    label_info, no_wrong_obj = objs_infomap2foDetections(objs_info, img_meta)
    sample.add_labels(dict(ground_truth=label_info))

    sample["xml_md5"] = md5sum(xml_path)

    if not os.path.exists(anno_path):
        logging.debug("{} do not have anno!".format(img_path))
        sample["chiebot_ID"] = "game_" + md5sum(
            img_path) if not os.path.basename(img_path).startswith(
                "game_") else os.path.basename(img_path)
        return sample

    with open(anno_path, 'r') as fr:
        anno = json.load(fr)
    sample["chiebot_ID"] = anno.get("ID", "game_" + md5sum(img_path))
    data_source = anno.get("data_source", None)
    data_source = [data_source] if isinstance(data_source,
                                              str) else data_source
    sample["data_source"] = data_source
    sample["img_quality"] = anno.get("img_quality", 0)
    sample["additions"] = anno.get("additions", None)

    return sample
