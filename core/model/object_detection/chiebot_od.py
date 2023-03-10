# -*- coding: utf-8 -*-
"""
@Author: captainsama
@Date: 2023-03-10 10:17:14
@LastEditors: captainsama tuanzhangsama@outlook.com
@LastEditTime: 2023-03-10 10:40:13
@FilePath: /dataset_manager/core/model/object_detection/chiebot_od.py
@Description:
"""
import grpc
from .proto import dldetection_pb2
from .proto import dldetection_pb2_grpc
from .base_detection import ProtoBaseDetection
from core.utils import img2base64
import cv2


class ChiebotObjectDetection(ProtoBaseDetection):
    def __init__(self,host,model_type: int = 1):
        self.host=host
        self.model_type = model_type

    def __enter__(self):
        channel_opt = [('grpc.max_send_message_length', 512 * 1024 * 1024), ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
        self.channel = grpc.insecure_channel(self.host,options =channel_opt)
        self.stub = dldetection_pb2_grpc.AiServiceStub(self.channel)
        return self

    def __exit__(self, exc_type, exc_value, trace):
        self.channel.close()
        return super().__exit__(exc_type, exc_value, trace)

    def __call__(self, img):
        if isinstance(img, str):
            img = cv2.imread(img, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        imgbase64 = img2base64(img)
        req = dldetection_pb2.DlRequest()
        req.imdata = imgbase64
        req.type = self.model_type
        response = self.stub.DlDetection(req)

        final_result = []

        for obj in response.results:
            final_result.append(
                (
                    obj.classid,
                    obj.score,
                    obj.rect.x / w,
                    obj.rect.y / h,
                    obj.rect.w / w,
                    obj.rect.h / h,
                )
            )

        return final_result
