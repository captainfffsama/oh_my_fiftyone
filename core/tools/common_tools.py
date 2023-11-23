# -*- coding: utf-8 -*-
"""
@Author: captainsama
@Date: 2023-03-08 15:10:57
@LastEditors: captainsama tuanzhangsama@outlook.com
@LastEditTime: 2023-03-08 15:18:15
@FilePath: /dataset_manager/core/tools/common_tools.py
@Description:
"""
import copy
from typing import Optional, Union, List,Tuple
from pprint import pprint
import weakref
from datetime import datetime
from functools import wraps
import time
import gc
from sklearn.model_selection import train_test_split
import os
from concurrent import futures
from collections import defaultdict
import random
import operator

import fiftyone as fo
import fiftyone.core.dataset as focd
import fiftyone.core.view as focv
import fiftyone.core.sample as focs
import fiftyone.brain as fob
from tqdm import tqdm
from PIL import Image
import piexif
import cv2
import numpy as np
import qdrant_client as qc
from fiftyone import ViewField as F

from core.utils import get_all_file_path, optimize_view, split_data_force, analytics_split
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
            字段使用参见 T.get_embedding

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
        with dataset.save_context() as context:
            with fo.ProgressBar(total=len(dataset),
                                start_msg="模型检测进度:",
                                complete_msg="检测完毕") as pb:
                with model as m:
                    for sample in dataset:
                        r=m.predict(sample.filepath)
                        sample[save_field] = r
                        context.save(sample)
                        pb.update(1)

    #TODO: 其他模型支持待补
    else:
        pass

    session = WEAK_CACHE.get("session", None)
    if session is not None:
        session.refresh()


def _infer(sample:Union[np.ndarray,str,focs.Sample,focs.SampleView], model):
    if isinstance(sample,focs.Sample) or isinstance(sample,focs.SampleView):
        sample=sample.filepath
    if isinstance(sample,str):
        sample=cv2.imread(sample,cv2.IMREAD_IGNORE_ORIENTATION|cv2.IMREAD_COLOR)
    result=model.embed(sample)
    return result

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

    Example:
        >>> # 使用ChiebotObjectDetection,假设服务器的地址为 http://127.0.0.1:52007
        >>> T.get_embedding(model_initargs={"host": "127.0.0.1:52007"})
        >>> # 若使用其他模型,比如使用自带的dinov2
        >>> import fiftyone.zoo as foz
        >>> model = foz.load_zoo_model("dinov2-vits14-torch", ensure_requirements=False)
        >>> T.get_embedding(model=model)
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
            logging.warning("model_initargs and model can not both None")
            return
        model = ChiebotObjectDetection(**model_initargs)

    with fo.ProgressBar(total=len(dataset),
                        start_msg="模型检测进度:",
                        complete_msg="检测完毕") as pb:

        with dataset.save_context() as context:
            with model as m:
                for sample in pb(dataset):
                    r=_infer(sample, m)
                    sample[save_field]=r
                    context.save(sample)
                # get_infer=lambda sample,m:(weakref.proxy(sample),_infer(sample,m))
                # with futures.ThreadPoolExecutor(2) as exec:
                #     tasks=(exec.submit(get_infer, sample, m) for sample in dataset)
                #     for task in pb(futures.as_completed(tasks)):
                #         r=task.result()
                #         print(r)
                        # sample[save_field]=r
                        # context.save(sample)

    session = WEAK_CACHE.get("session", None)
    if session is not None:
        session.refresh()



# TODO:添加不同相似度方法支持
@print_time_deco
def find_similar_img(
    image: Union[str, np.ndarray],
    model_initargs: Optional[dict] = None,
    model: Optional[ProtoBaseDetection] = None,
    dataset: Optional[Union[focd.Dataset,focv.DatasetView]]=None,
    qdrant_collection_name: Optional[str] = None,
    topk: int = 3,
    show: bool =True,
) -> List[Tuple[str,float]]:
    """从数据库中查找相似图片

    Args:
        image: Union[str,np.ndarray],
            待查图片

        model_initargs: (Optional[dict],optinal):
            用于初始化默认模型实例的参数,对于ChiebotObjectDetection就是模型类型
            比如:dict(host="127.0.0.1:52007")

        model (Optional[ProtoBaseDetection], optional):
            用于检测模型实例. Defaults to None.默认使用ChiebotObjectDetection

        dataset: Optional[Union[focd.Dataset,focv.DatasetView]]=None,
            在哪些数据中找,若不写,那就是session.dataset

        qdrant_collection_name : Optional[str] = None:
            qc数据库仓库名称,不写就默认是dataset.name + "_sim"

        topk: int = 3:
            取最相似的多少个图片

        show: bool = True:
            是否在浏览器上显示相似的图片并在终端显示结果

    Returns:
        List[Tuple[str,float]]: 相似图片完整路径列表和相似度分数

    Example:
        >>> # 若要嵌入到其他脚本中应用,建议取消wrap
        >>> find_simi_img=T.find_similar_img.__wrapper__

    """
    s: fo.Session = WEAK_CACHE.get("session", None)
    if s is None:
        logging.warning("no dataset in cache,do no thing")
        print("s is None")
        return []

    if dataset is None:
        dataset=s.dataset

    dataset_name = dataset.name if isinstance(dataset,focd.Dataset) else dataset.dataset_name
    qc_url = fob.brain_config.similarity_backends.get("qdrant", {}).get(
        "url", "127.0.0.1:6333")
    qc_client = qc.QdrantClient(url=qc_url)
    if qdrant_collection_name is None:
        qdrant_collection_name = dataset_name + "_sim"
    try:
        qc_client.get_collection(qdrant_collection_name)
    except Exception as e:
        logging.warning("{} not exist".format(qdrant_collection_name))
        print("collection {} not exist".format(qdrant_collection_name))
        print(e)
        return []

    if model is None:
        if model_initargs is None:
            logging.error("model or model_initarg can not be None both")
            return
        model = ChiebotObjectDetection(**model_initargs)

    with model as m:
        img_embed = _infer(image,m)

    # if img_embed is None:
    #     logging.error("generate img failed")
    #     print("generate img failed")
    #     return

    search_results = qc_client.search(
        collection_name=qdrant_collection_name,
        query_vector=img_embed,
        with_payload=True,
        limit=topk,
    )

    tmp = [(qdrant_point.payload["sample_id"], qdrant_point.score)
           for qdrant_point in search_results]
    tmp.sort(key=lambda x: x[1], reverse=True)
    if show:
        print("===========================================")
    similar_imgs_id = []
    file_socre_map={}
    for idx, x in enumerate(tmp):
        if show:
            print("{:3}---{} : {}".format(idx + 1, dataset[x[0]].filepath, x[1]))
        similar_imgs_id.append(x[0])
        file_socre_map[dataset[x[0]].filepath]=x[1]
    if show:
        print("===========================================")
    simi_imgs_dv:focv.DatasetView=dataset.select(similar_imgs_id, ordered=True)
    if show:
        s.view = simi_imgs_dv
        s.refresh()
    simi_imgs_fp=simi_imgs_dv.values("filepath")
    return [(x,file_socre_map[x]) for x in simi_imgs_fp]


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
            sample.set_field(KEY, tuple(content), validate=False, dynamic=True)
        else:
            sample.set_field(KEY, tuple(tags), validate=False, dynamic=True)


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
            sample.set_field(KEY, tuple(content), validate=False, dynamic=True)

@print_time_deco
def split_dataset(
    split_ratio: List[float],
    tags: Optional[str] = None,
    class_list: Optional[List[str]] = None,
    force_overwrite: bool = False,
    dataset: Optional[Union[fo.Dataset, fo.DatasetView]] = None
):
    """
    :param split_ratio: 数据划分比例,比例相加必须为1，可输入三个（则自动划分训练、验证、测试），也可输入两个（则自动划分训练、测试），可输入整数也可输入小数
    :param tags:  设定划分后的字段名称, 默认"auto"，即auto_train
                  字符串_train、字符串_val、字符串_test
    :param class_list: 按照给定的类别划分数据集，若class_list为None，则不考虑类别，直接暴力划分数据集
    :param force_overwrite: False. 当划分数据分配的字段与存在的字段发生冲突时，是否强制覆盖已存在的字段
                            True（请谨慎操作）, 直接清空一存在的冲突字段，将新划分后的数据写入该字段
    :param dataset: An optional Dataset or DatasetView object representing
                    the dataset on which the function will be applied.
    :return:
    """
    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is not None:
            dataset = s.dataset
        else:
            logging.warning("no dataset in cache,do no thing")
            return

    assert round(sum(split_ratio), 3) == 1, "the sum of split_ratio must be 1, 别把数据搞丢了"  # 0.7 + 0.1 = 0.899999
    assert isinstance(split_ratio, list), "输入的split_ratio不合法，应为List"
    assert len(split_ratio) <= 3, "len(split_ratio) > 3， 输入的很不河里"
    dataset = optimize_view(dataset)
    file_path = dataset.distinct('filepath')
    random.shuffle(file_path)
    dataset = dataset.select_by('filepath', file_path)
    dataset_temp = copy.deepcopy(dataset)
    split_train, split_val, split_test = [], [], []

    if class_list:
        # 分析数据集的各类别情况
        cls_info = defaultdict(int)
        for sample in tqdm(
            dataset,
            total=len(dataset),
            desc="数据集分析进度:",
            dynamic_ncols=True,
            colour="green",
        ):
            detections = sample["ground_truth"]["detections"]
            cur_list = []
            for det in detections:
                single_cls = det['label']
                if single_cls in class_list:
                    cls_info[single_cls] += 1

        cls_info = dict(sorted(cls_info.items(), key=operator.itemgetter(1)))
        cls_info_temp = copy.deepcopy(cls_info)
        visited_classes = set()
        while cls_info_temp:
            cls = list(cls_info_temp.items())[0][0]  # 类别数量最少的cls
            num = list(cls_info_temp.items())[0][1]
            print('当前类别为：{}， label数量为：{}'.format(cls, num))
            cls_data = dataset_temp.match(F('ground_truth.detections.label').contains([cls, '']))
            cls_path = cls_data.distinct('filepath')

            if not cls_path:
                visited_classes.add(cls)
                if cls in cls_info_temp:
                    del cls_info_temp[cls]
                continue

            cur_cls_info = analytics_split(cls_data, class_list)
            for key in cur_cls_info:
                if key in visited_classes:
                    continue
                cls_info[key] -= cur_cls_info[key]

            temp_train, temp_val, temp_test = split_data_force(cls_path, split_ratio)  # 单纯按类别划分，不考虑跨类别均衡情况下，运行时间可以接受
            total_split = len(temp_train) + len(temp_val) + len(temp_test)
            assert len(cls_path) == total_split, '!!!!! {}  !!!!!当前类别:{}，划分前：{}，划分后：{}'.format(len(cls_path), cls, len(cls_path), total_split)
            dataset_temp = dataset_temp.exclude_by('filepath', cls_path)

            split_train.extend(temp_train)
            split_val.extend(temp_val)
            split_test.extend(temp_test)
            cls_info_temp = dict(sorted(cls_info_temp.items(), key=lambda x: x[1]))  # 重新排序
            visited_classes.add(cls)
            del cls_info_temp[cls]
        for id, path in enumerate([split_train, split_val, split_test]):
            if id == 0:
                other_data = dataset_temp.exclude_by('filepath', path)
            else:
                other_data = other_data.exclude_by('filepath', path)
        other_path = other_data.distinct('filepath')
        # 对于不在cls_list中的类别，直接暴力划分数据
        temp_train, temp_val, temp_test = split_data_force(other_path, split_ratio)
        split_train.extend(temp_train)
        split_val.extend(temp_val)
        split_test.extend(temp_test)
    # 粗鄙之人： 直接暴力划分
    else:
        filepath = dataset.distinct('filepath')
        split_train, split_val, split_test = split_data_force(filepath, split_ratio)

    sample_tag = tags if tags else "auto"
    exists_tags = dataset.distinct('tags')
    print('划分之后的训练集:{}、验证集:{}、测试集:{}'.format(len(split_train), len(split_val), len(split_test)))
    if sample_tag + '_train' in exists_tags or sample_tag + '_test' in exists_tags or sample_tag + '_val' in exists_tags:
        print('---------------------------------------------', force_overwrite)
        if force_overwrite:
            print('给你一次反悔的机会，是否要强制删除并覆盖{}、{}、{}字段，是：y/Y, 否：输入其他任意字符'.format(sample_tag + '_train', sample_tag + '_val', sample_tag + '_test'))
            select = input()
            if select == 'y' or select == 'Y':
                dataset.untag_samples(sample_tag + '_train')
                dataset.untag_samples(sample_tag + '_val')
                dataset.untag_samples(sample_tag + '_test')
                print('已强制删除{}、{}、{} 字段'.format(sample_tag + '_train', sample_tag + '_val', sample_tag + '_test'))
            else:
                print('下次可不会再给你机会了！！！')
                return
        else:
            print('The target field already exists, please force_overwrite set to True')
            logging.warning("The target field already exists, please force_overwrite set to True")
            return

    train_data = dataset.select_by('filepath', split_train)
    train_data.tag_samples(sample_tag + '_train')
    if split_val:
        val_data = dataset.select_by('filepath', split_val)
        val_data.tag_samples(sample_tag + '_val')
    test_data = dataset.select_by('filepath', split_test)
    test_data.tag_samples(sample_tag + '_test')
    print('\033[1;34m数据划分成功，相应字段为{}、{}、{}\033[0m'.format(
        sample_tag + '_train', sample_tag + '_val', sample_tag + '_test'))