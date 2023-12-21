# -*- coding: utf-8 -*-
"""
@Author: captainsama
@Date: 2023-03-08 15:09:19
@LastEditors: captainsama tuanzhangsama@outlook.com
@LastEditTime: 2023-03-08 16:31:52
@FilePath: /dataset_manager/core/tools/dataset_opt.py
@Description:
"""
import os
import json
import time
import traceback
from typing import Optional, Union, List, Set
from datetime import datetime
from dataclasses import dataclass, field

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.validation import Validator
import qdrant_client as qc

import numpy as np
from tqdm import tqdm

import fiftyone as fo
import fiftyone.core.dataset as focd
import fiftyone.core.view as focv
import fiftyone.core.session as focs
import fiftyone.brain as fob
from fiftyone import ViewField as F

from core.utils import md5sum, get_all_file_path, optimize_view
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
        # TODO: 这里效率需要优化
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

    update_dataview=dataset.select_by("filepath",update_img_path_list)
    update_dataview.tag_samples(str(datetime.now().replace(microsecond=0)) + "update")

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

    dataset = optimize_view(dataset)

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
def clean_dataset(dataset: Optional[Union[fo.Dataset,
                                          fo.DatasetView]] = None, ):
    """清除数据库中实际文件已经不存在的样本

    Args:
        dataset (Optional[Union[fo.Dataset,fo.DatasetView]], optional): 和其他函数一样,默认是全局数据集. Defaults to None.
    """
    if dataset is None:
        s = WEAK_CACHE.get("session", None)
        if s is None:
            logging.warning("no dataset in cache,do no thing")
            return
        else:
            dataset = s.dataset
    dataset = optimize_view(dataset)

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
    dataset = optimize_view(dataset)
    if isinstance(dataset, focd.Dataset):
        qdrant_collection_name = dataset.name + "_sim"
    else:
        qdrant_collection_name = dataset.dataset_name + "_sim"
    if brain_key in dataset.list_brain_runs():
        previous_brain_run = dataset.load_brain_results(brain_key)
        if previous_brain_run:
            previous_brain_run.cleanup()
        qc_url = fob.brain_config.similarity_backends.get("qdrant", {}).get(
            "url", "127.0.0.1:6333")
        qc_client = qc.QdrantClient(url=qc_url)
        qc_client.delete_collection(qdrant_collection_name)
        dataset.delete_brain_run(brain_key)
    result = fob.compute_similarity(dataset,
                                    embeddings="embedding",
                                    backend="qdrant",
                                    brain_key=brain_key,
                                    metric="cosine",
                                    collection_name=qdrant_collection_name,
                                    **kwargs)
    s = WEAK_CACHE.get("session", None)
    if s is not None:
        s.refresh()
    return result


def _is_dup(method, search_score, score_thr) -> bool:
    if method == "euclidean":
        return search_score <= score_thr
    else:
        return search_score >= score_thr


@dataclass
class DupTmpResult:
    need_save_sample_ids: List[str] = field(default_factory=list)
    need_save_sample_simi_imgs: List[str] = field(default_factory=list)
    need_save_sample_simi_score: List[float] = field(default_factory=list)
    need_save_sample_simi_method: List[str] = field(default_factory=list)

    all_dup_51_ids: Set[str] = field(default_factory=set)


@print_time_deco
def duplicate_det(
    query_dataset: Optional[Union[focv.DatasetView, focd.Dataset]] = None,
    similar_thr: float = 0.985,
    check_thr: float = 0.955,
    similar_method: str = "cosine",
    key_dataset: Optional[Union[focv.DatasetView, focd.Dataset]] = None,
    query_have_done_ids: Optional[List[str]] = None,
    check_default_input: str = "y",
) -> List[str]:
    """
    注释是gpt写的,我懒
    Args:
    query_dataset (Optional[focv.DatasetView, focd.Dataset]): 要执行重复检测的数据集。默认为None。
    similar_thr (float): 将两个样本视为重复的相似度阈值。默认为0.985。
    check_thr (float): 将搜索结果视为潜在重复的分数阈值。默认为0.955。
    similar_method (str): 计算样本之间相似度的方法。必须是"cosine"、"dotproduct"或"euclidean"之一。默认为"cosine"。
    key_dataset (Optional[focv.DatasetView, focd.Dataset]): 用作重复检测的关键数据集。默认为None。
    query_have_done_ids (Optional[List[str]]): 已经处理过的查询样本的ID列表。默认为None。
    check_default_input: str= "y": 人工检查时的默认输入。默认为"y"。

    Returns:
    List[str]: 已经处理过的查询样本的ID列表。
    """
    s: focs.Session = WEAK_CACHE.get("session", None)
    if s is None:
        logging.error("no dataset in cache,do no thing")
        return
    if check_default_input not in {"y", "t"}:
        logging.warning(
            "check_default_input must be in y,t,check_default_input fix to y")
        check_default_input = "y"

    if similar_method not in {"cosine", "dotproduct", "euclidean"}:
        logging.warning(
            "similar_method must in cosine, dotproduct, euclidean,similar_method fix to cosine"
        )
        similar_method = "cosine"

    if query_dataset is not None and key_dataset is not None:
        query_dataset_name = query_dataset.dataset_name if isinstance(
            query_dataset, focv.DatasetView) else query_dataset.name
        key_dataset_name = key_dataset.dataset_name if isinstance(
            key_dataset, focv.DatasetView) else key_dataset.name
        if query_dataset_name != key_dataset_name:
            logging.warning(
                "query_dataset and key_dataset must be in the same dataset")
            return

    if query_dataset is None:
        CURRENT_DATASET: focd.Dataset = s.dataset
        query_dataset_ids = set(CURRENT_DATASET.values("id"))
    else:
        if isinstance(query_dataset,focv.DatasetView):
            CURRENT_DATASET: focd.Dataset = focd.load_dataset(
                query_dataset.dataset_name)
        else:
            CURRENT_DATASET: focd.Dataset = query_dataset
        query_dataset_ids = set(query_dataset.values("id"))

    print("current dataset is {}".format(CURRENT_DATASET.name))

    query_dataset_ids = query_dataset_ids - set(
        CURRENT_DATASET.match_tags("dup").values("id"))

    if key_dataset is None:
        key_dataset_ids = query_dataset_ids.copy()
    else:
        key_dataset_ids = set(
            key_dataset.exclude(
                key_dataset.match_tags("dup").values("id")).values("id"))

    # 处理存档
    if query_have_done_ids is None:
        query_have_done_ids = set([])
    else:
        query_have_done_ids = set(query_have_done_ids)
        query_dataset_ids = query_dataset_ids - query_have_done_ids

    query_dataset: focv.DatasetView = CURRENT_DATASET.select(query_dataset_ids)
    key_dataset: focv.DatasetView = CURRENT_DATASET.select(key_dataset_ids)
    tmp_d = query_dataset.match(F("embedding").is_null())
    if len(tmp_d):
        print(
            "query_dataset have some img no embedding,please use T.get_embedding to get embedding"
        )
        return list(query_have_done_ids)

    tmp_d = key_dataset.match(F("embedding").is_null())
    if len(tmp_d):
        print(
            "key_dataset have some img no embedding,please use T.get_embedding to get embedding"
        )
        return list(query_have_done_ids)

    qdrant_collection_name = query_dataset.dataset_name + "_dup_det"

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

    dup_tmp_r = DupTmpResult()

    dup_tmp_r.all_dup_51_ids.update(
        key_dataset.match_tags("dup").values("id") +
        query_dataset.match_tags("dup").values("id"))

    with fo.ProgressBar(total=len(query_dataset),
                        start_msg="样本检查重复进度:",
                        complete_msg="样本重复检查完毕") as pb:
        try:
            for idx, current_query_sample in enumerate(query_dataset):
                if (current_query_sample.id in dup_tmp_r.all_dup_51_ids):
                    query_have_done_ids.add(current_query_sample.id)
                    if not pb.complete:
                        pb.update(1)
                    continue
                if not current_query_sample.has_field("embedding"):
                    if not pb.complete:
                        pb.update(1)
                    print("{}样本没有embedding,跳过".format(
                        current_query_sample.filepath))
                    logging.warning("{}样本没有embedding,跳过".format(
                        current_query_sample.filepath))
                    continue

                current_query_feat: np.ndarray = current_query_sample.embedding
                search_results = qc_client.search(
                    collection_name=qdrant_collection_name,
                    query_vector=current_query_feat,
                    with_payload=True,
                    limit=100,
                    score_threshold=check_thr,
                )

                if not search_results:
                    query_have_done_ids.add(current_query_sample.id)
                    if not pb.complete:
                        pb.update(1)
                    continue

                need_check_samples_map = {}
                need_check_samples_info = []
                key_dup_info_map = {}
                for qdrant_point in search_results:
                    fiftyone_sid = qdrant_point.payload["sample_id"]
                    if (fiftyone_sid == current_query_sample.id
                            or fiftyone_sid in dup_tmp_r.all_dup_51_ids):
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

                    show_51_ids = [current_query_sample.id] + need_check_51_ids

                    time.sleep(0.5)
                    s.view = CURRENT_DATASET.select(show_51_ids, ordered=True)

                    t2 = prompt(
                        "\n 是否完成非重复标记?\n输入y将所有标记记为非重复,输入t将所有标记记为重复,输入e将所有标记记为非重复并退出 [y/t/e]:",
                        validator=valida,
                        completer=WordCompleter(["y", "t", "e"]),
                        default=check_default_input,
                    )
                    if t2 in ("y", "e"):
                        need_check_51_ids = set(need_check_51_ids)
                        sselected = set(s.selected)
                        dup_51_ids = list(need_check_51_ids - sselected)
                        for sid in dup_51_ids:
                            key_dup_info_map[sid] = need_check_samples_map[sid]

                        if current_query_sample.id not in sselected:
                            similar_sample_51_id = list(need_check_51_ids)[0]
                            dup_tmp_r.all_dup_51_ids.add(
                                current_query_sample.id)
                            dup_tmp_r.need_save_sample_ids.append(
                                current_query_sample.id)
                            dup_tmp_r.need_save_sample_simi_imgs.append(
                                key_dataset[similar_sample_51_id].filepath)
                            dup_tmp_r.need_save_sample_simi_score.append(
                                need_check_samples_map[similar_sample_51_id]
                                [1])
                            dup_tmp_r.need_save_sample_simi_method.append(
                                similar_method)
                    else:
                        sselected = set(s.selected)
                        dup_51_ids = sselected
                        if current_query_sample.id in sselected:
                            similar_sample_51_id = list(need_check_51_ids)[0]
                            dup_tmp_r.all_dup_51_ids.add(
                                current_query_sample.id)
                            dup_tmp_r.need_save_sample_ids.append(
                                current_query_sample.id)
                            dup_tmp_r.need_save_sample_simi_imgs.append(
                                key_dataset[similar_sample_51_id].filepath)
                            dup_tmp_r.need_save_sample_simi_score.append(
                                need_check_samples_map[similar_sample_51_id]
                                [1])
                            dup_tmp_r.need_save_sample_simi_method.append(
                                similar_method)

                            dup_51_ids.remove(current_query_sample.id)
                        for sid in dup_51_ids:
                            key_dup_info_map[sid] = need_check_samples_map[sid]

                    s.clear_view()

                key_sample_51ids = []
                key_sample_similar_img = [current_query_sample.filepath] * len(
                    key_dup_info_map.keys())
                key_sample_similar_method = [similar_method] * len(
                    key_dup_info_map.keys())
                key_sample_similar_score = []
                for k, v in key_dup_info_map.items():
                    key_sample_51ids.append(k)
                    key_sample_similar_score.append(v[1])

                if key_sample_51ids:
                    dup_tmp_r.all_dup_51_ids.update(key_sample_51ids)
                    dup_tmp_r.need_save_sample_ids.extend(key_sample_51ids)
                    dup_tmp_r.need_save_sample_simi_imgs.extend(
                        key_sample_similar_img)
                    dup_tmp_r.need_save_sample_simi_score.extend(
                        key_sample_similar_score)
                    dup_tmp_r.need_save_sample_simi_method.extend(
                        key_sample_similar_method)
                query_have_done_ids.add(current_query_sample.id)

                if t2 == "e":
                    if not pb.complete:
                        pb.update(1)
                    break

                if not pb.complete:
                    pb.update(1)

        except KeyboardInterrupt as e:
            pass
        except RuntimeError as e:
            traceback.print_exc(limit=-1)
            pass
        except Exception as e:
            traceback.print_exc(limit=-1)
            pass

        finally:
            # 保存结果
            print("将去重结果批量写入到数据库中...")
            # FIXME: 这里设计orderd 可能很吃内存
            if dup_tmp_r.need_save_sample_ids:
                dup_samples_view = CURRENT_DATASET.select(
                    dup_tmp_r.need_save_sample_ids, ordered=True)
                dup_samples_view.tag_samples("dup")

                if dup_tmp_r.need_save_sample_simi_imgs:
                    dup_samples_view.set_values(
                        "similar_img", dup_tmp_r.need_save_sample_simi_imgs)
                    dup_samples_view.set_values(
                        "similar_img_score",
                        dup_tmp_r.need_save_sample_simi_score)
                    dup_samples_view.set_values(
                        "similar_img_method",
                        dup_tmp_r.need_save_sample_simi_method)
                dup_samples_view.save()
            print("清理去重工作")
            similar_key_dealer.cleanup()
            if brain_key in key_dataset.list_brain_runs():
                key_dataset.delete_brain_run(brain_key)
            s.clear_view()
    return list(query_have_done_ids)


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
