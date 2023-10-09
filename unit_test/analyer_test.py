# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-10-07 10:48:16
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-10-09 10:03:55
@FilePath: /oh_my_fiftyone/unit_test/analyer_test.py
@Description:
'''
import unittest
import fiftyone.core.plots.plotly as fcpp
import core.tools as T
import fiftyone as fo
import plotly
import chart_studio.plotly as ppy


import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import re


class Analyer_datasetTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset=fo.load_dataset("test_f")
        return super().setUp()

    def test_pass(self):
        aa=T.DatasetAnalyer(self.dataset)
        # aa._build_figure()
        # aa._generate_class_graph("wcgz")
        aa.show()
        aa.export2excel("test.xlsx")



if __name__ == '__main__':
    unittest.main()