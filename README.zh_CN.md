[中文](./readme.md)   [English](./readme_EN.md)

[toc]

![logo](./doc/logo.png)
# Introduction

基于 fiftyone 写的一个管理数据的小东西

# requirements
详细见[requirements.txt](./requirements.txt)
## Optional requirements
- torch (安装torch的情况下可以使用自带模型或者是新加的嵌入模型)
- dash (数据集分析使用)
- diskcache (数据集分析使用)

详细见[requirements_full.txt](./requirements_full.txt)

# 最新进展
## 0.22.1
0.22.1 版本发布
### what's new
新增导入标签的方式,现在导入新数据可以选择 `new`,即完全使用新传入的标注作为最终标注
### fix bug
1. 修复导入数据时使用`merge`,nms方法可能造成的部分box丢失
## 0.22
0.22 版本发布
### what's new
新增 `T.DataAnalyer` 功能,在线分析目标检测数据集情况,使用示例如下:

```python
dataset=session.dataset.limit(10)
classes=["dog","cat"]
analyer=T.DataAnalyer(dataset,classes)
# 在线查看
analyer.show()
# 导出表格到 test.xlsx
analyer.export2excel("test.xlsx")
```

历次更新重点见[changelog.md](doc/changlog.md)

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
