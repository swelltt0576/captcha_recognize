# -*- coding: utf-8 -*-
# about captcha image
IMAGE_HEIGHT = 52
IMAGE_WIDTH = 120
# CHAR_SETS = 'abcdefghijklmnpqrstuvwxyz123456789ABCDEFGHIJKLMNPQRSTUVWXYZ'
CHAR_SETS = '0123456789'
CLASSES_NUM = len(CHAR_SETS)
CHARS_NUM = 4
# for train
RECORD_DIR = './data'
TRAIN_FILE = 'train.tfrecords'
VALID_FILE = 'valid.tfrecords'