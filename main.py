import fiftyone as fo
from core.importer import SGCCGameDatasetImporter
from IPython import embed

def main(data_dir):
    importer= SGCCGameDatasetImporter(dataset_dir=data_dir)
    dataset=fo.Dataset.from_importer(importer)
    session=fo.launch_app(dataset=dataset)
    embed()

if __name__ == '__main__':
    data_dir="/home/chiebotgpuhq/tmp_space/fif_test_data"
    main(data_dir)