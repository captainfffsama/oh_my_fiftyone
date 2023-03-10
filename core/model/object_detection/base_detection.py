# -*- coding: utf-8 -*-
'''
@Author: captainsama
@Date: 2023-03-10 10:24:57
@LastEditors: captainsama tuanzhangsama@outlook.com
@LastEditTime: 2023-03-10 11:53:58
@FilePath: /dataset_manager/core/model/object_detection/base_detection.py
@Description:
'''
from typing import Any
from abc import ABCMeta,abstractmethod

class ProtoBaseDetection(metaclass=ABCMeta):

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self,*args,**kwargs):
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass