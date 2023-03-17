# -*- coding:utf-8 -*-

"""
# File       : config_casRel.py
# Time       : 2023/2/28 21:36
# Author     : chenshuai
# version    : python 3.9
# Description: 配置文件
"""
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 1
LEARNING_RATE = 5e-5
EPOCH = 50
MODEL_SAVE_PATH = './save_model/'

tokenizer_path = './macbert_chinese_base'
bert_config_path = './macbert_chinese_base/config.json'
bert_model_path = './macbert_chinese_base'


train_data_path = '../data/input/duie/duie_train.json'
test_data_path = '../data/input/duie/duie_test.json'
dev_data_path = '../data/input/duie/duie_dev.json'
shema_path = '../data/input/duie/duie_schema.json'
rel_data_path = './data_file/rel_data.sv'


RELATION_SIZE = 48
BATCH_SIZE = 100
BERT_SIZE = 768

CLS_WEIGHT_COEF = [0.3, 1.0]
SUB_WEIGHT_COEF = 3

SUB_HEAD_BAR = 0.5
SUB_TAIL_BAR = 0.5
OBJ_HEAD_BAR = 0.5
OBJ_TAIL_BAR = 0.5

EPS = 1e-10

MODEL_DIR = './save_model/'

REL_PATH = '../data/output/rel.csv'