# -*- coding: utf-8 -*-
"""
@Author: captainsama
@Date: 2023-03-08 15:10:57
@LastEditors: captainsama tuanzhangsama@outlook.com
@LastEditTime: 2023-03-08 15:18:15
@FilePath: /dataset_manager/core/tools/common_tools.py
@Description:
"""

from typing import Optional, Union, List
from pprint import pprint
from datetime import datetime
from functools import wraps
import time

import os
import json
from concurrent import futures

import fiftyone as fo
import fiftyone.core.dataset as focd
import fiftyone.core.view as focv
import fiftyone.brain as fob
from tqdm import tqdm
from PIL import Image
import piexif
import cv2
import numpy as np
import qdrant_client as qc

from core.utils import get_sample_field, md5sum, get_all_file_path, optimize_view
from core.exporter.sgccgame_dataset_exporter import SGCCGameDatasetExporter
from core.logging import logging

from core.cache import WEAK_CACHE
from core.model import ProtoBaseDetection, ChiebotObjectDetection


def print_time_deco(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            end = time.perf_counter()
            print("\033[1;34m操作完成时间: {}, 操作耗时: {} 秒\033[0m".format(
                datetime.now(), end - start))
        return result

    return wrapper


@print_time_deco
def get_select_dv(txt_path: str = None,
                  ordered: bool = False) -> Optional[fo.DatasetView]:
    """返回被选中的数据的视图,若有txt就返回txt中的,没有就是浏览器中选中的
    Args:
        txt_path (Optional[str]):txt是一个记录了图片路径的文本文件
        ordered (bool, optional): 是否按照时间排序. Defaults to False.
            超级大数据的时候,选择使用ordered 会报 aggregate command document too large 的错误

    Returns:
        Optional[fo.DatasetView]: 返回被选中的数据的视图
    """

    session = WEAK_CACHE.get("session", None)
    if session is None:
        logging.warning("no dataset in cache,no thing export")
        return
    else:
        dataset: fo.Dataset = session.dataset
    if dataset and session:
        if txt_path is not None:
            if os.path.exists(txt_path):
                imgs_path = get_all_file_path(txt_path)
                return dataset.select_by("filepath",
                                         imgs_path,
                                         ordered=ordered)
        else:
            return dataset.select(session.selected, ordered=ordered)
    return None


@print_time_deco
def dataset_value2txt(
    value: str = "filepath",
    save_txt: Optional[str] = None,
    dataset: Optional[fo.DatasetView] = None,
):
    """将数据集的特定字段导入到txt

    Args:
        value (str, optional): 需要导出的字段. Defaults to "filepath".
        save_txt (Optional[str], optional): 需要保存的txt的路径,若为None就仅print出. Defaults to None.
        dataset (Optional[fo.DatasetView], optional): 需要导出字段的数据集,若为None就是 ``session.dataset``. Defaults to None.
    """
    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            dataset = s.dataset

    v = dataset.values(value)
    if save_txt:
        with open(save_txt, "w") as fw:
            for vv in v:
                fw.write(str(vv) + "\n")
    else:
        pprint(v)


@print_time_deco
def imgslist2dataview(imgslist: Union[str, List[str]],
                      dataset: Optional[fo.Dataset] = None,
                      ordered: bool = False) -> fo.DatasetView:
    """传入文件列表本身或者路径得到对应的dataview

    Args:
        imgslist (Union[str, List[str]]): 可以是一个记录了图片文件绝对路径的txt,或者图片目录或者是一个pythonlist
        dataset (Optional[fo.Dataset], optional): 同以前函数. Defaults to None.
        ordered (bool, optional): 是否按照时间排序. Defaults to False.
            超级大数据的时候,选择使用ordered 会报 aggregate command document too large 的错误

    Returns:
        fo.DatasetView: 图片list的dataview
    """
    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            dataset = s.dataset

    if isinstance(imgslist, str):
        imgslist = get_all_file_path(imgslist)

    return dataset.select_by("filepath", imgslist, ordered=ordered)


@print_time_deco
def check_dataset_exif(
    dataset: Optional[focd.Dataset] = None,
    clean_inplace: bool = False,
    log_path: Optional[str] = None,
    cv2_fix: bool = False,
) -> fo.DatasetView:
    """检查数据集中是否包含exif

    Args:
        dataset (Optional[focd.Dataset], optional):
            _description_. Defaults to None.

        clean_inplace (bool, optional):
            是否原地移除exif. Defaults to False.

        log_path (Optional[str], optional):
            若不为None,则将包含了exif的样本路径导出到txt. Defaults to None.

        cv2_fix (bool, optinal):
            在clean_inplace为True的情况下,若piexif报错,指示是否使用opencv读入读出. Defaults False,

    Returns:
        fo.DatasetView: 包含了exif的数据的dataview
    """
    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            dataset = s.dataset

    have_exif = []
    for sample in tqdm(
            dataset,
            desc="exif 检查进度:",
            dynamic_ncols=True,
            colour="green",
    ):
        img = Image.open(sample["filepath"])
        if "exif" in img.info:
            have_exif.append(sample["filepath"])
            if clean_inplace:
                try:
                    piexif.remove(sample["filepath"])
                except Exception as e:
                    logging.critical("{} remove piexif faild".format(
                        sample["filepath"]))
                    print("{} remove piexif faild".format(sample["filepath"]))
                    if cv2_fix and isinstance(e, piexif.InvalidImageDataError):
                        img = cv2.imread(
                            sample.filepath,
                            cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR,
                        )
                        cv2.imwrite(sample.filepath, img)

    if log_path:
        with open(log_path, "w") as fw:
            fw.writelines([x + "\n" for x in have_exif])

    print("Here is exif image path:")
    pprint(have_exif)

    return imgslist2dataview(have_exif, dataset)


@print_time_deco
def model_det(
    model_initargs: Optional[dict] = None,
    model: Optional[ProtoBaseDetection] = None,
    dataset: Optional[Union[fo.Dataset, fo.DatasetView]] = None,
    save_field: Optional[str] = "model_predict",
):
    """使用模型检测数据集,并将结果存到sample的model_predic 字段

    Args:
        model_initargs: (Optional[dict],optinal):
            用于初始化默认模型实例的参数,对于ChiebotObjectDetection就是模型类型

        model (Optional[ProtoBaseDetection], optional):
            用于检测模型实例. Defaults to None.默认使用ChiebotObjectDetection

        dataset (Optional[Union[fo.Dataset,fo.DatasetView]], optional):
            同之前. Defaults to None.

        save_field: Optional[str] = "model_predict":
            用来保存结果的字段.默认是Sample的model_predict 字段

    """
    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            dataset = s.dataset
    dataset = optimize_view(dataset)
    if model is None:
        if model_initargs is None:
            model_initargs = {}
        model = ChiebotObjectDetection(**model_initargs)

    if isinstance(model, ProtoBaseDetection):
        with fo.ProgressBar(total=len(dataset),
                            start_msg="模型检测进度:",
                            complete_msg="检测完毕") as pb:
            with model as m:
                deal_one = lambda s, mm: (s, mm.predict(s.filepath))
                with futures.ThreadPoolExecutor(10) as exec:
                    tasks = [
                        exec.submit(deal_one, sample, m) for sample in dataset
                    ]
                    for task in pb(futures.as_completed(tasks)):
                        sample, objs = task.result()

                        sample[save_field] = objs
                        sample.save()
    else:
        pass

    session = WEAK_CACHE.get("session", None)
    if session is not None:
        session.refresh()


@print_time_deco
def get_embedding(
    model_initargs: Optional[dict] = None,
    model: Optional[ProtoBaseDetection] = None,
    dataset: Optional[Union[fo.Dataset, fo.DatasetView]] = None,
    save_field: Optional[str] = "embedding",
):
    """使用模型检测数据集,并将结果存到sample的model_predic 字段

    Args:
        model_initargs: (Optional[dict],optinal):
            用于初始化默认模型实例的参数,对于ChiebotObjectDetection就是模型类型

        model (Optional[ProtoBaseDetection], optional):
            用于检测模型实例. Defaults to None.默认使用ChiebotObjectDetection

        dataset (Optional[Union[fo.Dataset,fo.DatasetView]], optional):
            同之前. Defaults to None.

        save_field: Optional[str] = "embedding":
            用来保存结果的字段.默认是Sample的 embedding 字段

    """
    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            dataset = s.dataset
    dataset = optimize_view(dataset)
    if model is None:
        if model_initargs is None:
            model_initargs = {}
        model = ChiebotObjectDetection(**model_initargs)

    if isinstance(model, ProtoBaseDetection):
        with fo.ProgressBar(total=len(dataset),
                            start_msg="模型检测进度:",
                            complete_msg="检测完毕") as pb:
            with model as m:
                deal_one = lambda s, mm: (s, mm.embed(s.filepath, norm=True))
                with futures.ThreadPoolExecutor(10) as exec:
                    tasks = [
                        exec.submit(deal_one, sample, m) for sample in dataset
                    ]
                    for task in pb(futures.as_completed(tasks)):
                        sample, objs = task.result()

                        sample[save_field] = objs
                        sample.save()
    else:
        pass

    session = WEAK_CACHE.get("session", None)
    if session is not None:
        session.refresh()


@print_time_deco
def find_similar_img(
    image: Union[str, np.ndarray],
    model_initargs: Optional[dict] = None,
    model: Optional[ProtoBaseDetection] = None,
    qdrant_collection_name: Optional[str] = None,
    topk: int = 3,
):
    """从数据库中查找相似图片

    Args:
        image: Union[str,np.ndarray],
            待查图片

        model_initargs: (Optional[dict],optinal):
            用于初始化默认模型实例的参数,对于ChiebotObjectDetection就是模型类型
            默认参数是dict(host="127.0.0.1:52007")

        model (Optional[ProtoBaseDetection], optional):
            用于检测模型实例. Defaults to None.默认使用ChiebotObjectDetection

        qdrant_collection_name : Optional[str] = None:
            qc数据库仓库名称,不写就默认是session.dataset.name + "_sim"

        topk: int = 3:
            取最相似的多少个图片

    """
    s: fo.Session = WEAK_CACHE.get("session", None)
    if s is None:
        logging.warning("no dataset in cache,do no thing")
        print("s is None")
        return
    qc_url = fob.brain_config.similarity_backends.get("qdrant", {}).get(
        "url", "127.0.0.1:6333")
    qc_client = qc.QdrantClient(url=qc_url)
    if qdrant_collection_name is None:
        qdrant_collection_name = s.dataset.name + "_sim"
    try:
        qc_client.get_collection(qdrant_collection_name)
    except Exception as e:
        logging.warning("{} not exist".format(qdrant_collection_name))
        print("collection {} not exist".format(qdrant_collection_name))
        print(e)
        return

    if model is None:
        if model_initargs is None:
            model_initargs = {"host": "127.0.0.1:52007"}
        model = ChiebotObjectDetection(**model_initargs)

    img_embed = None
    if isinstance(model, ProtoBaseDetection):
        with model as m:
            img_embed = m.embed(image, norm=True)
    else:
        pass
    if img_embed is None:
        logging.error("generate img failed")
        print("generate img failed")
        return

    search_results = qc_client.search(
        collection_name=qdrant_collection_name,
        query_vector=img_embed,
        with_payload=True,
        limit=topk,
    )

    tmp = [(qdrant_point.payload["sample_id"], qdrant_point.score)
           for qdrant_point in search_results]
    tmp.sort(key=lambda x: x[1], reverse=True)
    print("===========================================")
    similar_imgs_id = []
    for idx, x in enumerate(tmp):
        print("{:3}---{} : {}".format(idx + 1, s.dataset[x[0]].filepath, x[1]))
        similar_imgs_id.append(x[0])
    print("===========================================")
    s.view = s.dataset.select(similar_imgs_id, ordered=True)
    s.refresh()


@print_time_deco
def tag_chiebot_sample(
    tags: Union[str, List[str]],
    dataset: Optional[Union[fo.Dataset, fo.DatasetView]] = None,
):
    """
    更新chiebot_sample_tags,没有这个字段的样本会创建chiebot_sample_tags字段并更新

    Parameters:
    - tags: A string or a list of strings representing the tags to be applied to the function.
    - dataset: An optional Dataset or DatasetView object representing the dataset on which the function will be applied.

    Returns:
    None
    """

    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            dataset = s.dataset

    if isinstance(tags, str):
        tags = {tags}
    else:
        tags = set(tags)

    dataset = optimize_view(dataset)
    KEY = "chiebot_sample_tags"
    for sample in tqdm(
            dataset.iter_samples(autosave=True),
            total=len(dataset),
            desc="{} 更新进度:".format(KEY),
            dynamic_ncols=True,
            colour="green",
    ):
        if sample.has_field(KEY):
            content = sample.get_field(KEY)
            if content is None:
                content = set()
            else:
                content = set(content)
            content |= tags
            sample.set_field(KEY,
                             tuple(content),
                             validate=False,
                             dynamic=False)
        else:
            sample.set_field(KEY, tuple(tags), validate=False, dynamic=False)


@print_time_deco
def untag_chiebot_sample(
    tags: Union[str, List[str]],
    dataset: Optional[Union[fo.Dataset, fo.DatasetView]] = None,
):
    """
    删除chiebot_sample_tags

    Parameters:
    - tags: A string or a list of strings representing the tags to be applied to the function.
    - dataset: An optional Dataset or DatasetView object representing the dataset on which the function will be applied.

    Returns:
    None
    """

    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            dataset = s.dataset

    if isinstance(tags, str):
        tags = {tags}
    else:
        tags = set(tags)

    dataset = optimize_view(dataset)
    KEY = "chiebot_sample_tags"
    for sample in tqdm(
            dataset.iter_samples(autosave=True),
            total=len(dataset),
            desc="{} 更新进度:".format(KEY),
            dynamic_ncols=True,
            colour="green",
    ):
        if sample.has_field(KEY):
            content = sample.get_field(KEY)
            if content is None:
                content = set()
            else:
                content = set(content)
            content -= tags
            sample.set_field(KEY,
                             tuple(content),
                             validate=False,
                             dynamic=False)
