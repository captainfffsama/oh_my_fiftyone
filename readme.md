[中文](./readme.md)   [English](./readme_EN.md)

[toc]

# Introduction

基于 fiftyone 写的一个管理数据的小东西

# requirements

- fiftyone >=0.21.6
- ipython
- loguru
- pid
- piexif
- grpc >= 1.51.0
- grpc-tools>=1.51.0
- qdrant-client

# 需要注意的设定字段

## 数据集字段和样本字段

- `embedding`: 存放着样本嵌入
- `ground_truth`: 存放样本标注
- `model_predict`: 存放模型预测结果
- `chiebot_ID`: 比赛样本原始ID
- `xml_md5`: 当前样本的xml md5
- `img_quality`: 样本图片质量,暂未用上
- `data_source`: 样本数据来源
- `chiebot_sample_tags`: 类似sample tags,用来标志样本
- `additions`: 其他一些奇奇怪怪的信息

## brain_key

- `im_sim_qdrant`: 存放着使用 qdrant 计算数据的相似性

# 使用说明

参见[使用说明](./doc/user_guide.md)

# TODO

- [X] 记录额外信息
- [ ] 自定义后端proto来支持labelimg,labelme,labelhomo 之类的工具
- [X] 优化一些工具api的使用
- [X] rpc模型支持
- [X] 已选数据一键导出
- [ ] 多人操作问题
- [X] 目标检测多种导出格式支持
- [ ] 优化anno数据导入导出和原始的一致性
- [X] 以图搜图

# BUG

- [ ] io繁忙的情况  tqdm卡住
- [ ] 重新建立数据集时会话断开链接,需要重启整个程序
