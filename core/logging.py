# -*- coding: utf-8 -*-
"""
@Author: 198-server
@Date: 2023-02-27 15:09:54
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-10-08 16:31:19
@FilePath: /oh_my_fiftyone/core/logging.py
@Description:
"""
import os
from loguru import logger
from .cfg import LOG_DIR

logger.remove(handler_id=None)
logging_path = os.path.join(LOG_DIR, "oh_my_fiftyone_{time}.log")
logger.add(
    sink=logging_path,
    rotation="300 KB",  # 按文件大小切割日志
    retention="30 days",  # 只保留30天的日志
    encoding="utf-8",  # 编码
    level="DEBUG",
)
logging = logger
