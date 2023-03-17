# -*- coding:utf-8 -*-

"""
# File       : data_procressing.py
# Time       : 2023/3/2 21:50
# Author     : chenshuai
# version    : python 3.9
# Description: 数据处理
"""

import pandas as pd
from config_casRel import *
import json


def general_rel():
    """
    生成关系 (需要提前执行)
    :return: DataFrame
    """

    with open(shema_path, 'r', encoding='utf8') as f:
        rel_list = list()

        for line in f.readlines():
            rel_json = json.loads(line)
            rel_list.append(rel_json['predicate'])

        rel_dict = {value: index for index, value in enumerate(rel_list)}
        df = pd.DataFrame(rel_dict.items())
        df.to_csv(rel_data_path, header=None, index=None)


def load_rel():
    rel_data = pd.read_csv(rel_data_path, names=['rel', 'rel_id'])

    return rel_data['rel'].tolist(), dict(rel_data.values)


# if __name__ == '__main__':
#     general_rel()
