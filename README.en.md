
[中文](./readme.md)  [English](./readme_EN.md)

[toc]

![logo](./doc/logo.png)
# Introduction

A small tool for managing data based on fiftyone.

# requirements
See [requirements.txt](./requirements.txt) for details.
## Optional requirements
- torch (to use built-in models or newly added embedded models)
- dash (for dataset analysis)
- diskcache (for dataset analysis)

See [requirements_full.txt](./requirements_full.txt) for details.

# Latest Progress
Version 0.22 released, added `T.DataAnalyer` feature to analyze object detection datasets online. Example usage:

```python
dataset = session.dataset.limit(10)
classes = ["dog", "cat"]
analyer = T.DataAnalyer(dataset, classes)
# View online
analyer.show()
# Export table to test.xlsx
analyer.export2excel("test.xlsx")
``````
# Important fields to note

## Dataset fields and sample fields

- `embedding`: Stores sample embeddings.
- `ground_truth`: Stores sample annotations.
- `model_predict`: Stores model prediction results.
- `chiebot_ID`: Original ID of the sample for the competition.
- `xml_md5`: MD5 hash of the current sample's XML.
- `img_quality`: Quality of the sample image (not currently used).
- `data_source`: Source of the sample data.
- `chiebot_sample_tags`: Similar to sample tags, used to label samples.
- `additions`: Stores other miscellaneous information.

## brain_key

- `im_sim_qdrant`: Stores data similarity calculated using Qdrant.

# Usage Instructions

See [User Guide](./doc/user_guide.md) for usage instructions.


# TODO

- [X] Record additional information.
- [ ] Customize backend proto to support tools like labelimg, labelme, labelhomo.
- [X] Optimize the usage of some tool APIs.
- [X] Support RPC models.
- [X] One-click export of selected data.
- [ ] Multi-user operation issues.
- [X] Support multiple export formats for object detection.
- [ ] Optimize annotation data import/export and consistency with the original.
- [X] Image search by image.

# BUG

- [ ] tqdm gets stuck in busy IO situations.
- [ ] Session disconnects when rebuilding datasets, requiring a restart of the entire program