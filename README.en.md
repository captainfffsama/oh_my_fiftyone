
[中文](./README.md)   [English](./README.en.md)

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

# Latest Updates
## Version 0.22.1
Version 0.22.1 Released
### What's New
Added a new way to import labels. Now, when importing new data, you can choose 'new,' which means using the newly provided annotations as the final annotations.
### Bug Fixes
1. Fixed an issue that could cause the loss of some boxes when importing data using the 'merge' and 'nms' methods.

## Version 0.22
Version 0.22 Released
### What's New
Added the `T.DataAnalyzer` feature for online analysis of object detection datasets. Here's how to use it:

```python
dataset=session.dataset.limit(10)
classes=["dog","cat"]
analyzer=T.DataAnalyzer(dataset,classes)
# View online
analyzer.show()
# Export the table to test.xlsx
analyzer.export2excel("test.xlsx")
```
For a detailed history of updates, please refer to [changelog.md](doc/changlog.md).



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