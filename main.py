from functools import partial

import fiftyone as fo
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


def launch_dataset(dataset):
    session = fo.launch_app(dataset=dataset,
                            address="0.0.0.0",
                            remote=True,
                            auto=True)
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

    valida = Validator.from_callable(lambda x: x in exist_dataset,
                                     error_message="没有这个数据集")
    if exist_dataset:
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


def main():
    prompt_session = PromptSession()
    function_map = {
        "1": add_data2exsist_dataset,
        "2": check_exsist_dataset,
        "3": init_new_dataset
    }
    while True:
        main_vali = Validator.from_callable(partial(number_in_ranger,
                                                    min=1,
                                                    max=3),
                                            error_message="瞎输啥编号呢,用退格删了重输")
        main_win_select = prompt_session.prompt(
            HTML('''
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
===========================================
        请输入要做事情的编号:'''),
            completer=WordCompleter([str(i) for i in range(1, 4)]),
            validator=main_vali)
        function_map[main_win_select]()


if __name__ == '__main__':
    main()
