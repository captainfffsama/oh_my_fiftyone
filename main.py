import fiftyone as fo
from IPython import embed
from core.dataset_generator import generate_dataset
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter,PathCompleter
from prompt_toolkit import print_formatted_text as print
from prompt_toolkit.shortcuts import yes_no_dialog

from core.utils import timeblock


def main():
    prompt_session = PromptSession()
    while True:
        exist_dataset=fo.list_datasets()
        exist_dataset= exist_dataset if exist_dataset is not None else []
        exist_dataset.append("None")
        text = prompt_session.prompt('请输入数据集名称:',
                                     completer=WordCompleter(exist_dataset),
                                     complete_in_thread=True)
        if "None"==text:
            text = prompt_session.prompt('请输入数据集路径:',completer=PathCompleter(),complete_in_thread=True)
            t1=prompt_session.prompt('请输入数据集名称:')
            # text= "/home/chiebotgpuhq/tmp_space/fif_test_data"
            repr(t1)

            if t1 in fo.list_datasets():
                t2=yes_no_dialog(title="老实交代覆不覆盖数据库",text="{} 数据集已经存在,继续将覆盖已有的数据集,是否继续?")
                if not t2:
                    continue
            if not t1:
                t1=None
            with timeblock():
                dataset = generate_dataset(text,name=t1)
            print("dataset load done")
        else:
            if text not in fo.list_datasets():
                print("没有 {} 这个数据集,请选 None 然后输入数据集地址".format(text))
                continue
            else:
                dataset = fo.load_dataset(text)
        session = fo.launch_app(dataset=dataset,address="0.0.0.0",remote=True,auto=True)
        print(dataset.first().field_names)
        embed()
        session.close()

if __name__ == '__main__':
    main()
