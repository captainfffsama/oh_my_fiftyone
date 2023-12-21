# 0.22.2
0.22.2 版本发布
## what's new
1. 新增数据划分方法
2. 添加批量操作底层xml的方法(危险操作)

## fix bug
1. 优化 `find_similar_img`
2. 现在合并标签会尝试合并新数据的字段


# 0.22.1
0.22.1 版本发布
## what's new
新增导入标签的方式,现在导入新数据可以选择 `new`,即完全使用新传入的标注作为最终标注
## fix bug
1. 修复导入数据时使用`merge`,nms方法可能造成的部分box丢失
# 0.22
0.22 版本发布,新增 `T.DataAnalyer` 功能,在线分析目标检测数据集情况,使用示例如下:

```python
dataset=session.dataset.limit(10)
classes=["dog","cat"]
analyer=T.DataAnalyer(dataset,classes)
# 在线查看
analyer.show()
# 导出表格到 test.xlsx
analyer.export2excel("test.xlsx")
```