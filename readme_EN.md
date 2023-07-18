[中文](./readme.md)   [English](./readme_EN.md)

[toc]

# Introduction

This is a small tool based on fiftyone for managing data.

# Requirements

- fiftyone >=0.21.4
- ipython
- loguru
- pid
- piexif
- grpc >= 1.51.0
- grpc-tools>=1.51.0
- qdrant-client

# Important Field Settings

## Dataset Fields and Sample Fields

- `embedding`: Stores the embeddings of samples.
- `ground_truth`: Stores the ground truth labels of samples.
- `model_predict`: Stores the model's prediction results for samples.
- `chiebot_ID`: Original IDs of competition samples.
- `xml_md5`: MD5 hash of the current sample's XML.
- `img_quality`: Sample image quality (currently not used).
- `data_source`: Source of sample data.
- `chiebot_sample_tags`: Similar to sample tags, used to label samples.
- `additions`: Other miscellaneous information.

## brain_key

- `im_sim_qdrant`: Stores the similarity of data computed using qdrant.

# Instructions

See [User Guide](./doc/user_guide.md) for instructions on how to use this tool.

# TODO

- [X] Record additional information.
- [ ] Customize backend proto to support tools like labelimg, labelme, labelhomo, etc.
- [X] Optimize the use of some tool APIs.
- [X] Support RPC model.
- [X] One-click export of selected data.
- [ ] Multiple user operation issues.
- [X] Support multiple export formats for object detection.
- [ ] Optimize the consistency of annotation data import/export and original data.

# BUG

- [ ] Stuck tqdm in IO-intensive situations.
- [ ] Disconnection when rebuilding datasets, requiring the entire program to be restarted.