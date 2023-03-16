# -*- coding: utf-8 -*-
'''
@Author: captainsama
@Date: 2023-03-10 10:24:57
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-03-16 16:25:33
@FilePath: /dataset_manager/core/model/object_detection/base_detection.py
@Description:
'''
from typing import Any
from abc import ABCMeta,abstractmethod

import fiftyone.core.models as focm
class ProtoBaseDetection(metaclass=ABCMeta):

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self,*args,**kwargs):
        pass


class FOCMDefaultSetBase(focm.Model):
    def __init__(self) -> None:
        self._preprocess_flag=False

    @property
    def media_type(self):
        """The media type processed by the model.

        Supported values are "image" and "video".
        """
        return "image"

    @property
    def has_logits(self):
        """Whether this instance can generate logits for its predictions.

        This method returns ``False`` by default. Methods that can generate
        logits will override this via implementing the
        :class:`LogitsMixin` interface.
        """
        return False

    @property
    def has_embeddings(self):
        """Whether this instance can generate embeddings.

        This method returns ``False`` by default. Methods that can generate
        embeddings will override this via implementing the
        :class:`EmbeddingsMixin` interface.
        """
        return False

    @property
    def ragged_batches(self):
        """True/False whether :meth:`transforms` may return tensors of
        different sizes. If True, then passing ragged lists of data to
        :meth:`predict_all` is not allowed.
        """
        return False

    @property
    def transforms(self):
        """The preprocessing function that will/must be applied to each input
        before prediction, or ``None`` if no preprocessing is performed.
        """
        return None

    @property
    def preprocess(self):
        """Whether to apply :meth:`transforms` during inference (True) or to
        assume that they have already been applied (False).
        """
        return self._preprocess_flag

    @preprocess.setter
    def preprocess(self, value):
        self._preprocess_flag=value