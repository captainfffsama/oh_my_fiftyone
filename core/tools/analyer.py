# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-10-07 10:39:14
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-10-09 09:41:34
@FilePath: /oh_my_fiftyone/core/tools/analyer.py
@Description:
'''
from typing import Optional, Union, List, Dict, Tuple
from itertools import chain
from uuid import uuid4

import numpy as np
import fiftyone as fo
import fiftyone.core.view as focv
import fiftyone.core.session as focs
from fiftyone import ViewField as F

from dash import Dash, html, dash_table, dcc, callback, Input, Output
from dash.long_callback import DiskcacheLongCallbackManager

from core.cache import WEAK_CACHE
from core.cfg import DISK_CACHE_DIR
import diskcache

DISK_CACHE = diskcache.Cache(DISK_CACHE_DIR)
## Diskcache
launch_uid = uuid4()
_long_callback_manager = DiskcacheLongCallbackManager(
    DISK_CACHE,
    cache_by=[lambda: launch_uid],
    expire=60,
)

DASH_APP = Dash(name="Oh My Fiftyone",
                long_callback_manager=_long_callback_manager)
WEAK_CACHE["dash_app"] = DASH_APP

from core.logging import logging
from core.utils import optimize_view
import plotly.express as px
import pandas as pd


class DatasetAnalyer:
    """用于分析数据集

    Example:
    >>> analyer=T.DatasetAnalyer()
    >>> # 在线观看
    >>> analyer.show()
    >>> # 导出表格到test.xlsx
    >>> analyer.export2excel("test.xlsx")
    """

    def __init__(self,
                 dataset: Optional[Union[focv.DatasetView, fo.Dataset]] = None,
                 show_classes: Optional[List[str]] = None) -> None:
        if dataset is None:
            s: focs.Session = WEAK_CACHE.get("session", None)
            if s is None:
                logging.error("no dataset in cache,do no thing")
                print("no dataset in cache,do no thing")
                return
            dataset = s.dataset

        self.dash_app: Dash = WEAK_CACHE.get("dash_app", None)
        self.dash_app.title = u"Oh My Fiftyone 数据分析"

        if self.dash_app is None:
            logging.error("no dash app in cache,do no thing")
            print("no dash app in cache,do no thing")
            return

        self.dataset = dataset
        all_classes = self.dataset.distinct("ground_truth.detections.label")
        if show_classes is None:
            self._show_classes = all_classes
        else:
            self._show_classes = list(set(show_classes) & set(all_classes))
        self.pd_data, self.boxes_info = self._analy_dataset()

    def _build_figure(self):
        img_num_fig = px.histogram(self.pd_data, y=u"图片数", x=u"类别名")
        box_num_fig = px.histogram(self.pd_data, y=u"gt框数", x=u"类别名")
        self.dash_app.layout = html.Div([
            html.H1(children=u'数据集结果分析', style={'textAlign': 'center'}),
            dash_table.DataTable(data=self.pd_data.to_dict('records'),
                                 page_size=10),
            html.H1(children=u'图片数量图', style={'textAlign': 'center'}),
            dcc.Graph(figure=img_num_fig),
            html.H1(children=u'gt框数量图', style={'textAlign': 'center'}),
            dcc.Graph(figure=box_num_fig),
            dcc.Dropdown(self.pd_data[u"类别名"],
                         self._show_classes[0],
                         id='dropdown-selection'),
            html.Div(id="class_graph_space")
        ])

        # 设置回调,类似qt信号槽
        self.dash_app.long_callback(Output("class_graph_space", "children"),
                                    Input("dropdown-selection",
                                          "value"))(self._generate_class_graph)

    def _analy_dataset(self) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        pd_content = []
        # boxes shape: Nx4  4 is tlx,tly,w,h all values have been normed to 0~1
        boxes_info: Dict[str, np.ndarray] = {}

        with fo.ProgressBar(total=len(self._show_classes),
                            start_msg=u"数据集分析进度:",
                            complete_msg=u"分析完毕") as pb:
            for label_class in pb(self._show_classes):
                class_dataset = optimize_view(
                    self.dataset.filter_labels("ground_truth",
                                               F("label") == label_class))
                img_num = len(class_dataset)
                box_num = class_dataset.count("ground_truth.detections.label")
                box_avg_num_per_img = class_dataset.mean(
                    F("ground_truth.detections").length())
                box_std_num_per_img = class_dataset.std(
                    F("ground_truth.detections").length())
                _, box_max_num_per_img = class_dataset.bounds(
                    F("ground_truth.detections").length())

                rel_boxes = class_dataset.values(
                    F("ground_truth.detections.bounding_box"))
                rel_boxes = list(chain(*rel_boxes))
                boxes_np = np.asarray(rel_boxes)
                boxes_info[label_class] = boxes_np

                rbw_F, rbh_F = F("ground_truth.detections.bounding_box")[2], F(
                    "ground_truth.detections.bounding_box")[3]
                imw_F, imh_F = F("$metadata.width"), F("$metadata.height")
                rel_area_F = rbw_F * rbh_F
                abs_area_F = imw_F * imh_F * rel_area_F
                wh_ratio_F = (rbw_F * imw_F) / (rbh_F * imh_F)

                rel_area_avg = class_dataset.mean(rel_area_F)
                abs_area_avg = class_dataset.mean(abs_area_F)
                wh_ratio_avg = class_dataset.mean(wh_ratio_F)

                rel_area_std = class_dataset.std(rel_area_F)
                abs_area_std = class_dataset.std(abs_area_F)
                wh_ratio_std = class_dataset.std(wh_ratio_F)

                rel_area_min, rel_area_max = class_dataset.bounds(rel_area_F)
                abs_area_min, abs_area_max = class_dataset.bounds(abs_area_F)
                wh_ratio_min, wh_ratio_max = class_dataset.bounds(wh_ratio_F)

                pd_content.append({
                    u"类别名": label_class,
                    u"图片数": img_num,
                    u"gt框数": box_num,
                    u"每张图片平均gt框数": box_avg_num_per_img,
                    u"每张图片gt框数标准差": box_std_num_per_img,
                    u"单张图片最大gt框数": box_max_num_per_img,
                    u"gt框相对面积均值": rel_area_avg,
                    u"gt框相对面积标准差": rel_area_std,
                    u"gt框相对面积最小值": rel_area_min,
                    u"gt框相对面积最大值": rel_area_max,
                    u"gt框绝对面积均值": abs_area_avg,
                    u"gt框绝对面积标准差": abs_area_std,
                    u"gt框绝对面积最小值": abs_area_min,
                    u"gt框绝对面积最大值": abs_area_max,
                    u"gt框宽高比均值": wh_ratio_avg,
                    u"gt框宽高比标准差": wh_ratio_std,
                    u"gt框宽高比最小值": wh_ratio_min,
                    u"gt框宽高比最大值": wh_ratio_max,
                })
        pd_data = pd.DataFrame(pd_content, index=self._show_classes)
        return pd_data, boxes_info

    def _generate_class_graph(self, class_name) -> html.Div:
        class_dataset = optimize_view(
            self.dataset.filter_labels("ground_truth",
                                       F("label") == class_name))
        rtlx_F, rtly_F, rbw_F, rbh_F = F(
            "ground_truth.detections.bounding_box")[0], F(
                "ground_truth.detections.bounding_box")[1], F(
                    "ground_truth.detections.bounding_box")[2], F(
                        "ground_truth.detections.bounding_box")[3]
        imw_F, imh_F = F("$metadata.width"), F("$metadata.height")
        rel_area_F = rbw_F * rbh_F
        abs_area_F = imw_F * imh_F * rel_area_F
        wh_ratio_F = (rbw_F * imw_F) / (rbh_F * imh_F)

        rcx_F, rcy_F = rtlx_F + rbw_F / 2, rtly_F + rbh_F / 2
        # rel_h w 显示

        rbw, rbh, wh_ratio, rcx, rcy, rel_area, abs_area = class_dataset.values(
            [rbw_F, rbh_F, wh_ratio_F, rcx_F, rcy_F, rel_area_F, abs_area_F])

        rbw = list(chain(*rbw))
        rbh = list(chain(*rbh))
        wh_ratio = list(chain(*wh_ratio))
        rcx = list(chain(*rcx))
        rcy = list(chain(*rcy))
        rel_area = list(chain(*rel_area))
        abs_area = list(chain(*abs_area))

        class_pd_dataframe = pd.DataFrame({
            "rbw": rbw,
            "rbh": rbh,
            "wh_ratio": wh_ratio,
            "rcx": rcx,
            "rcy": rcy,
            "rel_area": rel_area,
            "abs_area": abs_area
        })

        rbwh_fig = px.density_heatmap(class_pd_dataframe,
                                      x="rbw",
                                      y="rbh",
                                      marginal_x="histogram",
                                      marginal_y="histogram")
        wh_ratio_fig = px.histogram(class_pd_dataframe,
                                    x="wh_ratio",
                                    nbins=100)
        rpos_fig = px.density_heatmap(class_pd_dataframe,
                                      x="rcx",
                                      y="rcy",
                                      marginal_x="histogram",
                                      marginal_y="histogram")
        rel_area = px.histogram(class_pd_dataframe, x="rel_area", nbins=100)
        abs_area = px.histogram(class_pd_dataframe, x="abs_area", nbins=100)

        div = html.Div(children=[
            html.Label(u"相对宽高热力图"),
            dcc.Graph(figure=rbwh_fig),
            html.Label(u"宽高比图"),
            dcc.Graph(figure=wh_ratio_fig),
            html.Label(u"目标框相对位置热力图"),
            dcc.Graph(figure=rpos_fig),
            html.Label(u"相对面积图"),
            dcc.Graph(figure=rel_area),
            html.Label(u"绝对面积图"),
            dcc.Graph(figure=abs_area),
        ])
        return div

    def show(self):
        self._build_figure()
        self.dash_app.run(host="0.0.0.0", port="51511")

    def export2excel(self, save_path):
        """
        Exports the data in the `pd_data` attribute to an Excel file at the specified `save_path`.

        Args:
            save_path: The path where the Excel file will be saved.
        """
        self.pd_data.to_excel(save_path, index=False)
