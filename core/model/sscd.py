# -*- coding: utf-8 -*-
"""
@Author: captainsama
@Date: 2023-03-10 10:17:14
@LastEditors: captainsama tuanzhangsama@outlook.com
@LastEditTime: 2023-03-10 10:40:13
@FilePath: /dataset_manager/core/model/object_detection/chiebot_od.py
@Description:
"""
from typing import Union, List
from concurrent import futures
import pathlib
import gc
from multiprocessing import Queue
from copy import deepcopy

import torch.multiprocessing as tmp
import torch
from torchvision import transforms
import numpy as np
import cv2
import fiftyone.core.labels as focl
import fiftyone.core.models as focm

from .object_detection import FOCMDefaultSetBase


def model_infer_process(sq: Queue, rq: Queue, ckpt_path: str, device: str):
    model = torch.jit.load(ckpt_path, map_location=device)
    transform = transforms.Compose([
        transforms.Resize(288),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    toTensor = transforms.ToTensor()
    model.eval()
    rq.put("model init done!")
    while True:
        sender: np.ndarray  = sq.get()
        if isinstance(sender,str):
            del model
            gc.collect()
            torch.cuda.empty_cache()
            return

        img_clone = deepcopy(sender)
        del sender
        img_clone = toTensor(img_clone).to(device)
        img_clone = transform(img_clone).unsqueeze(0)
        embedding: torch.Tensor = model(img_clone)[0, :]
        result = embedding.detach().cpu().numpy()
        rq.put(result)


# From: https://arxiv.org/abs/2202.10261
class SSCD(FOCMDefaultSetBase, focm.EmbeddingsMixin):

    def __init__(self, ckpt_path: str = None, device: str = "cuda:0"):
        if ckpt_path is None:
            save_dir = pathlib.Path(torch.hub.get_dir())
            save_dir = save_dir.parent.joinpath("fiftyone_models").joinpath(
                "sscd")
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            ckpt_path = save_dir.joinpath("sscd_imagenet_mixup.torchscript.pt")
        if not pathlib.Path(ckpt_path).exists():
            url = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_mixup.torchscript.pt"
            torch.hub.download_url_to_file(url, str(ckpt_path))
        self.ckpt_path = str(ckpt_path)
        self.device = device

        super().__init__()

    def __enter__(self):
        self.sq = Queue(5)
        self.rq = Queue(5)
        self.infer_process = tmp.Process(target=model_infer_process,
                                         args=(self.sq, self.rq,
                                               self.ckpt_path, self.device))
        self.infer_process.start()
        revice = self.rq.get()
        print(revice)
        del revice
        return self

    def __exit__(self, exc_type, exc_value, trace):
        self.sq.put("exit")
        self.infer_process.join()
        if self.infer_process.is_alive():
            try:
                self.infer_process.close()
            except Exception as e:
                print(e)
        self.sq.close()
        self.rq.close()
        return super().__exit__(exc_type, exc_value, trace)

    def predict(self, img: Union[np.ndarray, str]) -> np.ndarray:
        if isinstance(img, str):
            img = cv2.imread(img,
                             cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.sq.put(img)
        revice = self.rq.get()
        result = deepcopy(revice)
        del revice
        return result

    def predict_all(
            self, imgs: Union[np.ndarray, List[str]]) -> List[focl.Detections]:
        if isinstance(imgs, np.ndarray) and len(imgs.shape) < 4:
            imgs = [imgs]
        if isinstance(imgs, np.ndarray) and len(imgs.shape) == 4:
            batch = imgs.shape[0]
            imgs = np.split(imgs, batch, axis=0)

        with futures.ThreadPoolExecutor(min(len(imgs), 4)) as exec:
            results = [x for x in exec.map(self.predict, imgs)]

        return results

    @property
    def has_embeddings(self):
        """Whether this instance can generate embeddings.

        This method returns ``False`` by default. Methods that can generate
        embeddings will override this via implementing the
        :class:`EmbeddingsMixin` interface.
        """
        return True

    def get_embeddings(self):
        """Returns the embeddings generated by the last forward pass of the
        model.

        By convention, this method should always return an array whose first
        axis represents batch size (which will always be 1 when :meth:`predict`
        was last used).

        Returns:
            a numpy array containing the embedding(s)
        """
        return self.embed(self._last_embedding_img)

    def embed(self,
              img: Union[np.ndarray, str],
              norm: bool = True) -> np.ndarray:
        """Generates an embedding for the given data.

        Subclasses can override this method to increase efficiency, but, by
        default, this method simply calls :meth:`predict` and then returns
        :meth:`get_embeddings`.

        Args:
            img: the data. See :meth:`predict` for details
            norm: bool=False
                if true,embedding will be standardized and L2 normalized

        Returns:
            a numpy array containing the embedding
        """
        # pylint: disable=no-member
        self._last_embedding_img = img
        result = self.predict(img)
        if norm:
            result = (result - result.mean()) / result.std()
            return result / np.linalg.norm(result)
        return result

    def embed_all(self, imgs):
        """Generates embeddings for the given iterable of data.

        Subclasses can override this method to increase efficiency, but, by
        default, this method simply iterates over the data and applies
        :meth:`embed` to each.

        Args:
            args: an iterable of data. See :meth:`predict_all` for details

        Returns:
            a numpy array containing the embeddings stacked along axis 0
        """
        if isinstance(imgs, np.ndarray) and len(imgs.shape) < 4:
            imgs = [imgs]
        if isinstance(imgs, np.ndarray) and len(imgs.shape) == 4:
            batch = imgs.shape[0]
            imgs = np.split(imgs, batch, axis=0)

        with futures.ThreadPoolExecutor(min(len(imgs), 48)) as exec:
            results = [x for x in exec.map(self.embed, imgs)]

        return np.stack(results)
