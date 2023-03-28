[中文](./readme.md)   [English](./readme_EN.md)

# Introduction

基于 fiftyone 写的一个管理数据的小东西

# requirements

- fiftyone >=0.20.0
- ipython
- loguru
- pid
- piexif
- grpc == 1.51.0
- grpc-tools==1.51.0
- qdrant-client


# TODO

- [X] 记录额外信息
- [ ] 自定义后端proto来支持labelimg,labelme,labelhomo 之类的工具
- [X] 优化一些工具api的使用
- [X] rpc模型支持
- [X] 已选数据一键导出
- [ ] 多人操作问题
- [X] 目标检测多种导出格式支持

# BUG

- [ ] io繁忙的情况  tqdm卡住
