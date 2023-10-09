# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-10-08 16:07:20
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-10-08 16:25:33
@FilePath: /oh_my_fiftyone/core/cfg.py
@Description:
'''
import os
TMP_DIR="/tmp/oh_my_fiftyone"
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

LOG_DIR=os.path.join(TMP_DIR,"log")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

BAK_DIR=os.path.join(TMP_DIR,"bak")
if not os.path.exists(BAK_DIR):
    os.makedirs(BAK_DIR)
DISK_CACHE_DIR=os.path.join(TMP_DIR,"cache")
if not os.path.exists(DISK_CACHE_DIR):
    os.makedirs(DISK_CACHE_DIR)