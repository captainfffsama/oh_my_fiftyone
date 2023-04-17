import os
from functools import partial
from typing import Callable, Optional, Protocol
import time
from datetime import datetime
from collections import defaultdict

from pid.decorator import pidfile
from IPython import embed

import fiftyone as fo
import fiftyone.core.dataset as focd
import fiftyone.brain as fob
from fiftyone import ViewField as F
from fiftyone.utils.coco import COCODetectionDatasetExporter
from fiftyone.utils.yolo import YOLOv4DatasetExporter,YOLOv5DatasetExporter
from core.exporter.sgccgame_dataset_exporter import SGCCGameDatasetExporter

from prompt_toolkit import PromptSession, prompt
from prompt_toolkit.completion import WordCompleter, PathCompleter
from prompt_toolkit import print_formatted_text as print
from prompt_toolkit.shortcuts import yes_no_dialog
from prompt_toolkit.validation import Validator
from prompt_toolkit import print_formatted_text, HTML

from core.utils import timeblock,fol_det_nms
import core.tools as T
from core.cache import WEAK_CACHE
from core.data_preprocess import preprocess
from core.dataset_generator import generate_dataset, import_new_sample2exist_dataset
from core.model import ChiebotObjectDetection
from core.parse_label import parser_labels


class DatasetClass(Protocol):

    def __call__(self, s: fo.Session) -> focd.Dataset:
        pass


def launch_dataset(_d11: focd.Dataset):
    session = fo.launch_app(dataset=_d11,
                            address="0.0.0.0",
                            remote=True,
                            auto=True)
    WEAK_CACHE["session"] = session
    dataset: DatasetClass = lambda session=session: session.dataset
    embed(colors="linux")
    session.close()


def number_in_ranger(text: str, min=0, max=100):
    return text.isdigit() and min <= int(text) <= max


def add_data2exsist_dataset():
    prompt_session = PromptSession()
    text = prompt_session.prompt("请输入数据路径:",
                                 completer=PathCompleter(),
                                 complete_in_thread=True,
                                 validator=None)
    exist_dataset = fo.list_datasets()
    text = os.path.abspath(text)

    if exist_dataset:
        valida = Validator.from_callable(lambda x: x in exist_dataset,
                                         error_message="没有这个数据集")
        t1 = prompt_session.prompt(
            "请输入要导入的数据集名称:",
            validator=valida,
            completer=WordCompleter(exist_dataset),
            complete_in_thread=True,
        )

        valida1 = Validator.from_callable(lambda x: x in ("y", "n"),
                                          error_message="瞎选什么啊")
        t2 = prompt_session.prompt(
            "要不要把新数据原始数据考到已有数据文件夹? [y/n]:",
            validator=valida1,
            completer=WordCompleter(["y", "n"]),
        )
        dataset = fo.load_dataset(t1)
        if t2 == "n":
            with timeblock():
                new_dataset = generate_dataset(text, persistent=False)
                new_dataset.tag_samples(str(datetime.now()) + "import")

            dataset.merge_samples(new_dataset)
            dataset.save()
            print("dataset merge done")
        else:
            import_new_sample2exist_dataset(dataset, text)
            print("新数据导入完毕")
        launch_dataset(dataset)
    else:
        print_formatted_text(
            HTML("""
    <ansigreen>没有现成的数据集,没法追加数据集,请先创建数据集</ansigreen>
                                  """))
        time.sleep(1)


def check_exsist_dataset():
    prompt_session = PromptSession()
    exist_dataset = fo.list_datasets()
    if exist_dataset:
        valida = Validator.from_callable(lambda x: x in exist_dataset,
                                         error_message="没有这个数据集")
        text = prompt_session.prompt(
            "请输入需要查询的数据集名称,按Tab补全选:",
            completer=WordCompleter(exist_dataset),
            complete_in_thread=True,
            validator=valida,
        )
        dataset = fo.load_dataset(text)
        launch_dataset(dataset)
    else:
        print_formatted_text(
            HTML("""
    <ansigreen>没有现成的数据集</ansigreen>
                                  """))
        time.sleep(1)


def init_new_dataset():
    prompt_session = PromptSession()
    text = prompt_session.prompt("请输入数据集路径:",
                                 completer=PathCompleter(),
                                 complete_in_thread=True,
                                 validator=None)
    t1 = prompt_session.prompt("请输入新导入的数据集名称:", validator=None)
    # text= "/home/chiebotgpuhq/tmp_space/fif_test_data"
    text = os.path.abspath(text)
    if t1 in fo.list_datasets():
        t2 = yes_no_dialog(
            title="老实交代覆不覆盖数据库",
            text="{} 数据集已经存在,继续将覆盖已有的数据集,是否继续?".format(t1)).run()
        if not t2:
            return
    if not t1:
        t1 = None
    with timeblock():
        dataset = generate_dataset(text, name=t1)
    print("dataset load done")
    launch_dataset(dataset)


def delete_exsist_dataset():
    prompt_session = PromptSession()
    exist_dataset = fo.list_datasets()
    if exist_dataset:
        valida = Validator.from_callable(lambda x: x in exist_dataset,
                                         error_message="没有这个数据集")
        text = prompt_session.prompt(
            "请输入需要删除的数据集名称,按Tab补全选:",
            completer=WordCompleter(exist_dataset),
            complete_in_thread=True,
            validator=valida,
        )

        t2 = yes_no_dialog(title="你真的要删库跑路吗?你最好想想你在做什么!",
                           text="确定删除{}数据库吗?".format(text)).run()
        if t2:
            focd.delete_dataset(text)
            print_formatted_text(
                HTML("""
        <red>删除{}数据库成功!!</red>
                                    """.format(text)))
            time.sleep(1)
    else:
        print_formatted_text(
            HTML("""
    <ansigreen>没有现成的数据集</ansigreen>
                                  """))
        time.sleep(1)


def preprocess_data():
    prompt_session = PromptSession()
    dataset_dir = prompt_session.prompt("请输入样本路径:",
                                        completer=PathCompleter(),
                                        complete_in_thread=True,
                                        validator=None)

    dataset_dir = os.path.abspath(dataset_dir)

    save_dir = prompt_session.prompt("请输入保存路径:",
                                     completer=PathCompleter(),
                                     complete_in_thread=True,
                                     validator=None)

    save_dir = os.path.abspath(save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    valida1 = Validator.from_callable(lambda x: x in ("y", "n"),
                                      error_message="瞎选什么啊")
    t2 = prompt_session.prompt("是否进行md5重命名? [y/n]:",
                               validator=valida1,
                               completer=WordCompleter(["y", "n"]))

    rename = False
    prefix = ""

    if t2 == "y":
        rename = True
        t3 = prompt("请输入重命名前缀:", validator=None)
        prefix = t3

    convert2jpg = False
    t4 = prompt_session.prompt("是否全部转换为jpg? [y/n]:",
                               validator=valida1,
                               completer=WordCompleter(["y", "n"]))
    if t4 == "y":
        convert2jpg = True

    t5 = prompt(
        "记录额外信息的json路径:",
        completer=PathCompleter(),
        complete_in_thread=True,
        validator=None,
    )

    if not (os.path.exists(t5) and os.path.isfile(t5)):
        print("{} 不是有效路径,将不会生成anno".format(t5))
        t5 = None
    preprocess(dataset_dir, save_dir, rename, convert2jpg, prefix, t5)


def merge_label():

    win_show = HTML("""
===========================================
请保证输入的待处理样本标签目录中子目录是按照每份标签
进行划分,即如下:
- input dir
     |---1(labels part1)
     |         |----1.xml
     |         |----2.xml
     |         |---- ....
     |---2(labels part2)
     _....
其中coco的标签文件必须叫labels.json,voc的
标签文件后缀名xml必须小写,yolov4 记录class的文件
须为obj.names,yolov5记录class的文件须叫dataset.yaml

图片后缀名须是jpg

所有标签有关文件须和图片在同一级
===========================================
        请输入待处理的标签目录们的父目录:""")

    prompt_session = PromptSession()
    labels_part_dir = prompt_session.prompt(win_show,
                                            completer=PathCompleter(),
                                            complete_in_thread=True,
                                            validator=None)

    labels_part_dir = os.path.abspath(labels_part_dir)

    save_dir= prompt_session.prompt("合并结果的保存目录:",
                                    completer=PathCompleter(),
                                    complete_in_thread=True,
                                    validator=None)
    save_dir= os.path.abspath(save_dir)
    valida = Validator.from_callable(lambda x: x in ("voc", "coco", "yolov4","yolov5"),
                                     error_message="瞎选什么啊")
    type_c = WordCompleter(["voc", "coco", "yolov4","yolov5"])
    save_label_type= prompt("合并结果的标签格式:",
                           completer=type_c,
                           validator=valida,
                           default="voc")


    img_path_dict={}
    label_dict=defaultdict(list)

    for folder in os.listdir(labels_part_dir):
        i=os.path.join(labels_part_dir,folder)
        if os.path.isdir(i):
            result=parser_labels(folder)
            for k,v in result.items():
                label_dict[k].extend(v)
                img_path_dict[k]=os.path.join(i,k+".jpg")

    FORMAT_CLASS_MAP={
        "voc": SGCCGameDatasetExporter,
        "coco": COCODetectionDatasetExporter,
        "yolov4": YOLOv4DatasetExporter,
        "yolov5": YOLOv5DatasetExporter,
    }


    exporter =FORMAT_CLASS_MAP[save_label_type](export_dir=save_dir,)
    with exporter:
        with fo.ProgressBar(
                total=len(label_dict.keys), start_msg="标签文件合并进度:", complete_msg="合并完毕"
            ) as pb:
            for k,v in label_dict.items():
                objs=fol_det_nms(v,iou_thr=0.7)
                img_path=img_path_dict[k]
                metadata=fo.ImageMetadata.build_for(img_path)
                exporter.export_sample(img_path,objs, metadata=metadata)



@pidfile(pidname="dataset_manager")
def main():
    prompt_session = PromptSession()
    function_map = {
        "1": add_data2exsist_dataset,
        "2": check_exsist_dataset,
        "3": init_new_dataset,
        "4": delete_exsist_dataset,
        "5": preprocess_data,
        "6": merge_label,
    }
    while True:
        main_vali = Validator.from_callable(
            partial(number_in_ranger, min=1, max=len(function_map.keys())),
            error_message="瞎输啥编号呢,用退格删了重输",
        )
        main_win_show = HTML("""
===========================================
⠀⠀⠀⠀⠀⠀⠀⠀⠀⡴⠞⠉⢉⣭⣿⣿⠿⣳⣤⠴⠖⠛⣛⣿⣿⡷⠖⣶⣤⡀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⣼⠁⢀⣶⢻⡟⠿⠋⣴⠿⢻⣧⡴⠟⠋⠿⠛⠠⠾⢛⣵⣿⠀⠀⠀⠀
⣼⣿⡿⢶⣄⠀⢀⡇⢀⡿⠁⠈⠀⠀⣀⣉⣀⠘⣿⠀⠀⣀⣀⠀⠀⠀⠛⡹⠋⠀⠀⠀⠀
⣭⣤⡈⢑⣼⣻⣿⣧⡌⠁⠀⢀⣴⠟⠋⠉⠉⠛⣿⣴⠟⠋⠙⠻⣦⡰⣞⠁⢀⣤⣦⣤⠀
⠀⠀⣰⢫⣾⠋⣽⠟⠑⠛⢠⡟⠁⠀⠀⠀⠀⠀⠈⢻⡄⠀⠀⠀⠘⣷⡈⠻⣍⠤⢤⣌⣀
⢀⡞⣡⡌⠁⠀⠀⠀⠀⢀⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⢿⡀⠀⠀⠀⠸⣇⠀⢾⣷⢤⣬⣉
⡞⣼⣿⣤⣄⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⣿⠀⠸⣿⣇⠈⠻
⢰⣿⡿⢹⠃⠀⣠⠤⠶⣼⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⣿⠀⠀⣿⠛⡄⠀
⠈⠉⠁⠀⠀⠀⡟⡀⠀⠈⡗⠲⠶⠦⢤⣤⣤⣄⣀⣀⣸⣧⣤⣤⠤⠤⣿⣀⡀⠉⣼⡇⠀
⣿⣴⣴⡆⠀⠀⠻⣄⠀⠀⠡⠀⠀⠀⠈⠛⠋⠀⠀⠀⡈⠀⠻⠟⠀⢀⠋⠉⠙⢷⡿⡇⠀
⣻⡿⠏⠁⠀⠀⢠⡟⠀⠀⠀⠣⡀⠀⠀⠀⠀⠀⢀⣄⠀⠀⠀⠀⢀⠈⠀⢀⣀⡾⣴⠃⠀
⢿⠛⠀⠀⠀⠀⢸⠁⠀⠀⠀⠀⠈⠢⠄⣀⠠⠼⣁⠀⡱⠤⠤⠐⠁⠀⠀⣸⠋⢻⡟⠀⠀
⠈⢧⣀⣤⣶⡄⠘⣆⠀⠀⠀⠀⠀⠀⠀⢀⣤⠖⠛⠻⣄⠀⠀⠀⢀⣠⡾⠋⢀⡞⠀⠀⠀
⠀⠀⠻⣿⣿⡇⠀⠈⠓⢦⣤⣤⣤⡤⠞⠉⠀⠀⠀⠀⠈⠛⠒⠚⢩⡅⣠⡴⠋⠀⠀⠀⠀
⠀⠀⠀⠈⠻⢧⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⣻⠿⠋⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠉⠓⠶⣤⣄⣀⡀⠀⠀⠀⠀⠀⢀⣀⣠⡴⠖⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀

            你想对数据集做些什么？
1. 添加新的数据到已有数据集   6. 合并标注
2. 查看已有数据集
3. 建立新的数据集
4. 删除已有数据集
5. 处理数据
===========================================
        请输入要做事情的编号:""")

        main_win_select = prompt_session.prompt(
            main_win_show,
            completer=WordCompleter(function_map.keys()),
            validator=main_vali,
        )
        try:
            function_map[main_win_select]()
        except KeyboardInterrupt as e:
            continue


if __name__ == "__main__":
    main()
