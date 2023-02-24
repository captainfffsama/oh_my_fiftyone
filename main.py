import fiftyone as fo
from IPython import embed
from core.dataset_generator import generate_dataset
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter,PathCompleter
from prompt_toolkit import print_formatted_text as print


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
            # text= "/home/chiebotgpuhq/tmp_space/fif_test_data"
            dataset = generate_dataset(text)
            print("dataset load done")
        else:
            if text not in fo.list_datasets():
                print("没有 {} 这个数据集".fotmat(text))
                continue
            else:
                dataset = fo.load_dataset("data_tmp")
        session = fo.launch_app(dataset=dataset)
        embed()

if __name__ == '__main__':
    main()