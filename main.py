from functools import partial
import time

import fiftyone as fo
import fiftyone.core.dataset as focd
from IPython import embed
from core.dataset_generator import generate_dataset
from prompt_toolkit import PromptSession, prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter, PathCompleter
from prompt_toolkit import print_formatted_text as print
from prompt_toolkit.shortcuts import yes_no_dialog
from prompt_toolkit.validation import Validator
from prompt_toolkit import print_formatted_text, HTML

from core.utils import timeblock
import core.tools as T
from core.cache import WEAK_CACHE


def launch_dataset(dataset):
    session = fo.launch_app(dataset=dataset,
                            address="0.0.0.0",
                            remote=True,
                            auto=True)
    WEAK_CACHE["dataset"]=dataset
    WEAK_CACHE["session"]=session
    embed()
    session.close()


def number_in_ranger(text: str, min=0, max=100):
    return text.isdigit() and min <= int(text) <= max


def add_data2exsist_dataset():
    prompt_session = PromptSession()
    text = prompt_session.prompt('请输入数据路径:',
                                 completer=PathCompleter(),
                                 complete_in_thread=True,
                                 validator=None)
    exist_dataset = fo.list_datasets()

    if exist_dataset:
        valida = Validator.from_callable(lambda x: x in exist_dataset,
                                         error_message="没有这个数据集")
        t1 = prompt_session.prompt('请输入要导入的数据集名称:',
                                   validator=valida,
                                   completer=WordCompleter(exist_dataset),
                                   complete_in_thread=True)
        with timeblock():
            new_dataset = generate_dataset(text, persistent=False)

        dataset = fo.load_dataset(t1)
        dataset.merge_samples(new_dataset)
        dataset.save()
        print("dataset merge done")
        launch_dataset(dataset)
    else:
        print_formatted_text(
            HTML('''
    <ansigreen>没有现成的数据集,没法追加数据集,请先创建数据集</ansigreen>
                                  '''))
        time.sleep(1)


def check_exsist_dataset():
    prompt_session = PromptSession()
    exist_dataset = fo.list_datasets()
    if exist_dataset:
        valida = Validator.from_callable(lambda x: x in exist_dataset,
                                         error_message="没有这个数据集")
        text = prompt_session.prompt('请输入需要查询的数据集名称,按Tab补全选:',
                                     completer=WordCompleter(exist_dataset),
                                     complete_in_thread=True,
                                     validator=valida)
        dataset = fo.load_dataset(text)
        launch_dataset(dataset)
    else:
        print_formatted_text(
            HTML('''
    <ansigreen>没有现成的数据集</ansigreen>
                                  '''))
        time.sleep(1)


def init_new_dataset():
    prompt_session = PromptSession()
    text = prompt_session.prompt('请输入数据集路径:',
                                 completer=PathCompleter(),
                                 complete_in_thread=True,
                                 validator=None)
    t1 = prompt_session.prompt('请输入新导入的数据集名称:', validator=None)
    # text= "/home/chiebotgpuhq/tmp_space/fif_test_data"
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
        text = prompt_session.prompt('请输入需要删除的数据集名称,按Tab补全选:',
                                     completer=WordCompleter(exist_dataset),
                                     complete_in_thread=True,
                                     validator=valida)

        t2 = yes_no_dialog(title="你真的要删库跑路吗?你最好想想你在做什么!",
                           text="确定删除{}数据库吗?".format(text)).run()
        if t2:
            focd.delete_dataset(text)
            print_formatted_text(
                HTML('''
        <red>删除{}数据库成功!!</red>
                                    '''.format(text)))
            time.sleep(1)
    else:
        print_formatted_text(
            HTML('''
    <ansigreen>没有现成的数据集</ansigreen>
                                  '''))
        time.sleep(1)


def main():
    prompt_session = PromptSession()
    function_map = {
        "1": add_data2exsist_dataset,
        "2": check_exsist_dataset,
        "3": init_new_dataset,
        "4": delete_exsist_dataset,
    }
    while True:
        main_vali = Validator.from_callable(partial(number_in_ranger,
                                                    min=1,
                                                    max=len(
                                                        function_map.keys())),
                                            error_message="瞎输啥编号呢,用退格删了重输")
        main_win_show = HTML('''
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
        1. 添加新的数据到已有数据集
        2. 查看已有数据集
        3. 建立新的数据集
        4. 删除已有数据集
===========================================
        请输入要做事情的编号:''')

        main_win_select = prompt_session.prompt(main_win_show,
                                                completer=WordCompleter(
                                                    function_map.keys()),
                                                validator=main_vali)
        try:
            function_map[main_win_select]()
        except KeyboardInterrupt as e:
            continue


if __name__ == '__main__':
    main()
