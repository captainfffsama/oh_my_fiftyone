# -*- coding: utf-8 -*-
'''
@Author: captainsama
@Date: 2023-03-09 15:34:02
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-09-28 13:25:14
@FilePath: /oh_my_fiftyone/core/model/__init__.py
@Description:
'''

from .object_detection import ChiebotObjectDetection,ProtoBaseDetection
try:
    import torch
    from .sscd import SSCD
except ModuleNotFoundError:
    print("torch is not installed, SSCD cannot be loaded.")

# __all__=["ChiebotObjectDetection"]