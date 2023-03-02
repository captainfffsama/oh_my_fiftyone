# -*- coding: utf-8 -*-
'''
@Author: 198-server
@Date: 2023-02-27 15:09:54
@LastEditors: captainsama tuanzhangsama@outlook.com
@LastEditTime: 2023-03-02 09:49:41
@FilePath: /dataset_manager/core/logging.py
@Description:
'''
import datetime
from loguru import logger

logger.remove(handler_id=None)
logging_path='/tmp/dataset_manager_{}.log'.format(datetime.date.today())
logger.add(logging_path,enqueue=True)
logging=logger


# import logging

# class levelFilter(logging.Filter):
#     def filter(self, record):
#         if record.levelno < logging.WARNING:
#             return True
#         return False

# logger = logging.getLogger()
# log_formatter=logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
# fh=logging.FileHandler(filename='/tmp/dataset_manager_{}.log'.format(datetime.date.today()),mode='a')
# fh.setLevel(logging.WARNING)
# fh.setFormatter(log_formatter)

# ch=logging.StreamHandler()
# ch.setLevel(logging.INFO)
# ch.addFilter(levelFilter())

# logger.addHandler(fh)
# logger.addHandler(ch)
