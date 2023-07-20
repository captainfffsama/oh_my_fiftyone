# -*- coding: utf-8 -*-
"""
@Author: captainsama
@Date: 2023-03-08 15:09:19
@LastEditors: captainsama tuanzhangsama@outlook.com
@LastEditTime: 2023-03-08 16:31:52
@FilePath: /dataset_manager/core/tools/dataset_opt.py
@Description:
"""
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter, PathCompleter
from prompt_toolkit.validation import Validator
import qdrant_client as qc
from qdrant_client.models import PointIdsList
from qdrant_client.http.exceptions import UnexpectedResponse

import ipdb
from typing import Optional, Union, List, Callable, Tuple, Sequence, Iterable
from pprint import pprint
from datetime import datetime
from copy import deepcopy

import os
import json
from concurrent import futures

import fiftyone as fo
import fiftyone.core.dataset as focd
import fiftyone.core.session as focs
import fiftyone.brain as fob
from sklearn.metrics import pairwise_distances
import numpy as np
from tqdm import tqdm

from core.utils import get_sample_field, md5sum, get_all_file_path, timeblock
from core.exporter.sgccgame_dataset_exporter import SGCCGameDatasetExporter
from core.logging import logging

from core.cache import WEAK_CACHE
from core.importer import parse_sample_info, generate_sgcc_sample
from .common_tools import imgslist2dataview, print_time_deco


@print_time_deco
def update_dataset(
    dataset: Optional[focd.Dataset] = None,
    update_imgs_asbase: bool = True,
    sample_path_list: Optional[List[str]] = None,
):
    """更新数据集

    Args:
        dataset (Optional[focd.Dataset], optional):
            若dataset参数为None,那么将使用缓存引用中的dataset,这个dataset通常是全局的dataset

        update_imgs_asbase (bool) =True:
            若为False,更新根据数据集来,遍历数据集,更新其中xml发生变化的样本,该情况下,新的数据
            将不会被添加到数据集中.
            若为True,更新将根据样本文件来,样本文件由参数 ``sample_path_list``来确定,
            遍历样本文件,其中和数据集中记录不一样的或者没有的将被更新进数据集
            若 ``sample_path_list`` 没指定,那么样本文件列表为数据集所在文件夹的样本文件.

        sample_path_list (Optional[List[str]]):
            若为None,将从待更新的数据集所在文件夹的样本文件开始遍历,否则将根据提供的样本文件列表开始遍历
    """

    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            dataset = s.dataset
    update_img_path_list = []
    if update_imgs_asbase:
        if sample_path_list:
            imgs_path = sample_path_list
        else:
            dataset_dir = dataset.info.get(
                "dataset_dir",
                os.path.split(dataset.first().filepath)[0])
            imgs_path = get_all_file_path(
                dataset_dir,
                filter_=(
                    ".jpg",
                    ".JPG",
                    ".png",
                    ".PNG",
                    ".bmp",
                    ".BMP",
                    ".jpeg",
                    ".JPEG",
                ),
            )
        with dataset.save_context() as context:
            for img_path in tqdm(
                    imgs_path,
                    desc="数据集更新进度:",
                    dynamic_ncols=True,
                    colour="green",
            ):
                if img_path in dataset:
                    sample = dataset[img_path]
                    xml_path = os.path.splitext(sample.filepath)[0] + ".xml"
                    if not os.path.exists(xml_path):
                        sample.clear_field("ground_truth")
                        continue
                    xml_md5 = md5sum(xml_path)
                    if sample.has_field("xml_md5"):
                        if sample.get_field("xml_md5") != xml_md5:
                            img_meta, label_info, anno_dict = parse_sample_info(
                                sample.filepath)
                            sample.update_fields(anno_dict)
                            sample.update_fields({
                                "metadata": img_meta,
                                "ground_truth": label_info,
                                "xml_md5": xml_md5,
                            })
                            update_img_path_list.append(sample.filepath)
                        context.save(sample)
                else:
                    dataset.add_sample(generate_sgcc_sample(img_path))
        dataset.save()
    else:
        for sample in dataset.iter_samples(progress=True,
                                           autosave=True,
                                           batch_size=0.2):
            xml_path = os.path.splitext(sample.filepath)[0] + ".xml"
            if not os.path.exists(xml_path):
                sample.clear_field("ground_truth")
                continue
            xml_md5 = md5sum(xml_path)
            if sample.has_field("xml_md5"):
                if sample.get_field("xml_md5") != xml_md5:
                    img_meta, label_info, anno_dict = parse_sample_info(
                        sample.filepath)
                    sample.update_fields(anno_dict)
                    sample.update_fields({
                        "metadata": img_meta,
                        "ground_truth": label_info,
                        "xml_md5": xml_md5,
                    })

                    update_img_path_list.append(sample.filepath)

    # NOTE: 注意这里解除装饰器
    update_dataview = imgslist2dataview.__wrapped__(update_img_path_list,
                                                    dataset)
    update_dataview.tag_samples(str(datetime.now()) + "update")

    session = WEAK_CACHE.get("session", None)
    if session is not None:
        session.refresh()


@print_time_deco
def add_dataset_fields_by_txt(
    txt_path: Union[str, List[str]],
    fields_dict: Union[str, dict],
    dataset: Optional[focd.Dataset] = None,
):
    """通过txt给特定数据集添加字段,txt中不在数据集的数据将被跳过

    Args:
        txt_path (Union[str,List[str]]): 记录的图片路径的txt,或者一个列表
        fields_dict (Union[str, dict]): 可以是一个json或者一个dict
        dataset (Optional[focd.Dataset], optional): 和其他函数一样,默认是全局数据集. Defaults to None.
    """
    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            dataset = s.dataset

    if isinstance(txt_path, str):
        imgs_path = get_all_file_path(
            txt_path,
            filter_=(".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP", ".jpeg",
                     ".JPEG"),
        )
    else:
        imgs_path = txt_path

    if isinstance(fields_dict, str):
        with open(fields_dict, "r") as fr:
            fields_dict = json.load(fr)

    for sample in tqdm(
            dataset.select_by("filepath",
                              imgs_path).iter_samples(autosave=True),
            total=len(imgs_path),
            desc="字段更新进度:",
            dynamic_ncols=True,
            colour="green",
    ):
        for k, v in fields_dict.items():
            sample.set_field(k, v)

    session = WEAK_CACHE.get("session", None)
    if session is not None:
        session.refresh()


@print_time_deco
def clean_dataset(dataset: Optional[focd.Dataset] = None, ):
    """清除数据库中实际文件已经不存在的样本

    Args:
        dataset (Optional[focd.Dataset], optional): 和其他函数一样,默认是全局数据集. Defaults to None.
    """
    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            dataset = s.dataset

    need_del = []

    for sample in tqdm(
            dataset,
            total=len(dataset),
            desc="数据集检查进度:",
            dynamic_ncols=True,
            colour="green",
    ):
        if not os.path.exists(sample["filepath"]):
            need_del.append(sample["id"])

    dataset.delete_samples(need_del)
    dataset.save()

    session = WEAK_CACHE.get("session", None)
    if session is not None:
        session.refresh()


@print_time_deco
def generate_qdrant_idx(dataset: Optional[focd.Dataset] = None,
                        brain_key="im_sim_qdrant",
                        **kwargs):
    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            dataset = s.dataset
    if brain_key in dataset.list_brain_runs():
        previous_brain_run = dataset.load_brain_results(brain_key)
        if previous_brain_run:
            previous_brain_run.cleanup()
        # if isinstance(previous_brain_run,
        #               fob.internal.core.qdrant.QdrantSimilarityIndex):
        #     collections_name = [
        #         i.name for i in
        #         previous_brain_run.client.get_collections().collections
        #     ]
        #     print("将清除所有qdrant collection")
        #     for i in collections_name:
        #         previous_brain_run.client.delete_collection(i)
        dataset.delete_brain_run(brain_key)
    if isinstance(dataset, focd.Dataset):
        qdrant_collection_name = dataset.name + "_sim"
    else:
        qdrant_collection_name = dataset.dataset_name + "_sim"
    result = fob.compute_similarity(dataset,
                                    embeddings="embedding",
                                    backend="qdrant",
                                    brain_key=brain_key,
                                    metric="cosine",
                                    collection_name=qdrant_collection_name,
                                    **kwargs)
    s.refresh()
    return result


def _generate_dup_info(dataset: focd.Dataset) -> Tuple[str, List, Iterable]:
    if isinstance(dataset, focd.Dataset):
        qdrant_collection_name = dataset.name + "_dup_det"
    else:
        qdrant_collection_name = dataset.dataset_name + "_dup_det"

    imgs_id = dataset.values("id")
    imgs_id_iter = iter(imgs_id)
    return qdrant_collection_name, imgs_id, imgs_id_iter


def _is_dup(method, search_score, score_thr) -> bool:
    if method == "euclidean":
        return search_score <= score_thr
    else:
        return search_score >= score_thr


@print_time_deco
def duplicate_det(
    query_dataset: Optional[focd.Dataset] = None,
    similar_thr: float = 0.985,
    check_thr: float = 0.955,
    similar_method: str = "cosine",
    key_dataset: Optional[focd.Dataset] = None,
    query_have_done_ids: Optional[List[str]] = None,
) -> List[str]:
    """
    注释是gpt写的,我懒
    Args:
    query_dataset (Optional[focd.Dataset]): 要执行重复检测的数据集。默认为None。
    similar_thr (float): 将两个样本视为重复的相似度阈值。默认为0.985。
    check_thr (float): 将搜索结果视为潜在重复的分数阈值。默认为0.955。
    similar_method (str): 计算样本之间相似度的方法。必须是"cosine"、"dotproduct"或"euclidean"之一。默认为"cosine"。
    key_dataset (Optional[focd.Dataset]): 用作重复检测的关键数据集。默认为None。
    query_have_done_ids (Optional[List[str]]): 已经处理过的查询样本的ID列表。默认为None。

    Returns:
    List[str]: 已经处理过的查询样本的ID列表。
    """
    s: Optional[focs.Session] = WEAK_CACHE.get("session", None)
    assert similar_method in (
        "cosine",
        "dotproduct",
        "euclidean",
    ), "similar method must be in {}".format(
        ("cosine", "dotproduct", "euclidean"))
    if query_dataset is None:
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            query_dataset = s.dataset
    if query_have_done_ids is None:
        query_have_done_ids = set([])
    else:
        query_dataset = query_dataset.exclude(query_have_done_ids)
        query_have_done_ids = set(query_have_done_ids)

    query_dataset = query_dataset.exclude(
        query_dataset.match_tags("dup").values("id"))

    qdrant_collection_name, query_imgs_id, query_imgs_id_iter = _generate_dup_info(
        query_dataset)

    if key_dataset is None:
        key_dataset = query_dataset
    else:
        key_dataset = key_dataset.exclude(
            key_dataset.match_tags("dup").values("id"))

    _, query_imgs_id, query_imgs_id_iter = _generate_dup_info(query_dataset)

    qdrant_collection_name, key_imgs_id, key_imgs_id_iter = _generate_dup_info(
        key_dataset)

    print("建立临时索引中,等着吧...")
    brain_key = "qdrant_dup_det_brain"
    if brain_key in key_dataset.list_brain_runs():
        previous_brain_run = key_dataset.load_brain_results(brain_key)
        if previous_brain_run:
            previous_brain_run.cleanup()
        key_dataset.delete_brain_run(brain_key)
    similar_key_dealer = fob.compute_similarity(
        key_dataset,
        embeddings="embedding",
        backend="qdrant",
        brain_key=brain_key,
        metric=similar_method,
        collection_name=qdrant_collection_name,
    )
    qc_client: qc.QdrantClient = similar_key_dealer.client
    print("临时索引建立完毕")

    valida = Validator.from_callable(lambda x: x in ("y", "t", "e"),
                                     error_message="瞎选什么啊")

    with fo.ProgressBar(total=len(query_imgs_id),
                        start_msg="样本检查重复进度:",
                        complete_msg="样本重复检查完毕") as pb:
        try:
            all_dup_51_sample_id = set(
                key_dataset.match_tags("dup").values("id") +
                query_dataset.match_tags("dup").values("id"))
            while True:
                current_query = next(query_imgs_id_iter)
                if ("dup" in query_dataset[current_query].tags
                        or current_query in query_have_done_ids):
                    query_have_done_ids.add(current_query)
                    if not pb.complete:
                        pb.update(1)
                    continue
                current_query_feat: np.ndarray = query_dataset[
                    current_query].embedding
                search_results = qc_client.search(
                    collection_name=qdrant_collection_name,
                    query_vector=current_query_feat,
                    with_payload=True,
                    limit=100,
                    score_threshold=check_thr,
                )

                if not search_results:
                    query_have_done_ids.add(current_query)
                    if not pb.complete:
                        pb.update(1)
                    continue

                need_check_samples_map = {}
                need_check_samples_info = []
                key_dup_info_map = {}
                for qdrant_point in search_results:
                    fiftyone_sid = qdrant_point.payload["sample_id"]
                    if (fiftyone_sid == current_query
                            or fiftyone_sid in all_dup_51_sample_id):
                        continue
                    if _is_dup(similar_method, qdrant_point.score,
                               similar_thr):
                        key_dup_info_map[fiftyone_sid] = (
                            qdrant_point.id,
                            qdrant_point.score,
                        )
                    else:
                        need_check_samples_map[fiftyone_sid] = (
                            qdrant_point.id,
                            qdrant_point.score,
                        )
                        need_check_samples_info.append(
                            (fiftyone_sid, qdrant_point.score))

                t2 = ""
                if need_check_samples_info:
                    need_check_samples_info.sort(
                        key=lambda x: x[1],
                        reverse=(similar_method != "euclidean"))
                    need_check_51_ids = [x[0] for x in need_check_samples_info]

                    s.view = query_dataset.select(current_query).concat(
                        key_dataset.select(need_check_51_ids, ordered=True))
                    s.refresh()


                    t2 = prompt(
                        "\n 是否完成非重复标记? \n输入y将所有标记记为非重复,输入t将所有标记记为重复,输入e将所有标记记为非重复并退出 [y/t/e]:",
                        validator=valida,
                        completer=WordCompleter(["y", "t", "e"]),
                        default="y",
                    )
                    if t2 in ("y", "e"):
                        need_check_51_ids = set(need_check_51_ids)
                        sselected = set(s.selected)
                        dup_51_ids = list(need_check_51_ids - sselected)
                        for sid in dup_51_ids:
                            key_dup_info_map[sid] = need_check_samples_map[sid]

                        if current_query not in sselected:
                            all_dup_51_sample_id.add(current_query)
                            query_dataset.select(current_query).tag_samples(
                                "dup")
                            similar_sample_51_id = list(need_check_51_ids)[0]

                            query_dataset.select(current_query).set_values(
                                "similar_img",
                                [key_dataset[similar_sample_51_id].filepath],
                            )
                            query_dataset.select(current_query).set_values(
                                "similar_img_score",
                                [
                                    need_check_samples_map[
                                        similar_sample_51_id][1]
                                ],
                            )
                            query_dataset.select(current_query).set_values(
                                "similar_img_method", [similar_method])
                    else:
                        sselected = set(s.selected)
                        dup_51_ids = sselected
                        if current_query in sselected:
                            all_dup_51_sample_id.add(current_query)
                            query_dataset.select(current_query).tag_samples(
                                "dup")
                            similar_sample_51_id = list(need_check_51_ids)[0]

                            query_dataset.select(current_query).set_values(
                                "similar_img",
                                [key_dataset[similar_sample_51_id].filepath],
                            )
                            query_dataset.select(current_query).set_values(
                                "similar_img_score",
                                [
                                    need_check_samples_map[
                                        similar_sample_51_id][1]
                                ],
                            )
                            query_dataset.select(current_query).set_values(
                                "similar_img_method", [similar_method])
                            dup_51_ids.remove(current_query)

                        for sid in dup_51_ids:
                            key_dup_info_map[sid] = need_check_samples_map[sid]

                    s.clear_view()
                # dup_sampel_qdrant_ids = []
                key_sample_51ids = []
                key_sample_similar_img = [
                    query_dataset[current_query].filepath
                ] * len(key_dup_info_map.keys())
                key_sample_similar_method = [similar_method] * len(
                    key_dup_info_map.keys())
                key_sample_similar_score = []
                for k, v in key_dup_info_map.items():
                    key_sample_51ids.append(k)
                    key_sample_similar_score.append(v[1])
                    all_dup_51_sample_id.add(k)
                    # dup_sampel_qdrant_ids.append(v[0])

                if key_sample_51ids:
                    dataset_part = key_dataset.select(key_sample_51ids)
                    dataset_part.tag_samples("dup")
                    dataset_part.set_values("similar_img",
                                            key_sample_similar_img)
                    dataset_part.set_values("similar_img_score",
                                            key_sample_similar_score)
                    dataset_part.set_values("similar_img_method",
                                            key_sample_similar_method)

                # result=qc_client.delete(
                #     qdrant_collection_name,
                #     wait=True,
                #     points_selector=dup_sampel_qdrant_ids)
                query_have_done_ids.add(current_query)

                if t2 == "e":
                    if not pb.complete:
                        pb.update(1)
                    break

                if not pb.complete:
                    pb.update(1)

        except StopIteration as e:
            pass

        except KeyboardInterrupt as e:
            pass
        except RuntimeError as e:
            print(e)
            pass
        except Exception as e:
            print(e)
            pass

        finally:
            similar_key_dealer.cleanup()
            if brain_key in key_dataset.list_brain_runs():
                key_dataset.delete_brain_run(brain_key)
            s.refresh()
            return list(query_have_done_ids)

    # similar_key_dealer.cleanup()
    # if brain_key in key_dataset.list_brain_runs():
    #     key_dataset.delete_brain_run(brain_key)
    # s.refresh()
    # return query_have_done_ids


@print_time_deco
def clean_all_brain_qdrant():
    """
    Cleans all brain runs and qdrant in the current dataset.
    """
    s = WEAK_CACHE.get("session", None)
    if s is None:
        logging.warning("no dataset in cache,do no thing")
        return

    dataset: fo.Dataset = s.dataset
    dataset.delete_brain_runs()
    qc_client = qc.QdrantClient("localhost", port=6333)
    qc_collection_names = qc_client.get_collections()
    qc_client.delete_collection(qc_collection_names)
    s.refresh()
