# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-02-23 09:48:44
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-02-23 12:57:05
@FilePath: /dataset_manager/lib/importer/sgccgame_dataset_importer.py
@Description:
'''

import os
from typing import Optional, Tuple, List
import logging

import fiftyone as fo
import fiftyone.utils.data as foud
import fiftyone.core.metadata as fom
import fiftyone.core.labels as fol

from core.utils import get_all_file_path, parse_xml_info, parse_img_metadata, normalization_xyxy


class SGCCGameDatasetImporter(foud.LabeledImageDatasetImporter):

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

    def __next__(
        self
    ) -> Tuple[str, Optional[fom.ImageMetadata], Optional[fol.Detections]]:
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
        label_info=objs_infomap2foDetections(
            objs_info, img_meta)


        return current_img_path, img_meta, label_info
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
        return fol.Detections

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
                logging.warning("{} have wrong obj".format(
                    img_metadata.img_path))
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
