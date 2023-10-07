[中文](./readme.md)   [English](./readme_EN.md)

[toc]

![logo](./doc/logo.png)
# Introduction

A small tool for managing data based on FiftyOne.

# Requirements

- FiftyOne >= 0.22.0
- ipython
- loguru
- pid
- piexif
- grpc >= 1.51.0
- grpc-tools >= 1.51.0
- qdrant-client

## Optional Requirements

torch

When torch is installed, you can use built-in models or newly added embedding models.

# Important Configuration Fields

## Dataset Fields and Sample Fields

- `embedding`: Stores sample embeddings.
- `ground_truth`: Stores sample annotations.
- `model_predict`: Stores model prediction results.
- `chiebot_ID`: Original ID for competition samples.
- `xml_md5`: XML MD5 for the current sample.
- `img_quality`: Sample image quality, currently not in use.
- `data_source`: Sample data source.
- `chiebot_sample_tags`: Similar to sample tags, used to label samples.
- `additions`: Some other miscellaneous information.

## brain_key

- `im_sim_qdrant`: Stores data similarity calculated using Qdrant.

# Usage Instructions

Refer to the [User Guide](./doc/user_guide.md).

# TODO

- [X] Record additional information.
- [ ] Customize the backend proto to support tools like labelimg, labelme, labelhomo, etc.
- [X] Optimize the usage of some tools APIs.
- [X] Support RPC models.
- [X] Export selected data with one click.
- [ ] Address multi-user operation issues.
- [X] Support multiple export formats for object detection.
- [ ] Optimize data import/export and ensure consistency with the original data.
- [X] Image-based search.

# BUG

- [ ] In busy I/O situations, tqdm may hang.
- [ ] When rebuilding a dataset, the session disconnects and requires a complete program restart.
