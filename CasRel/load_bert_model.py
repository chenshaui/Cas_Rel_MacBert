# -*- coding:utf-8 -*-

"""
# File       : load_bert_model.py
# Time       : 2023/2/28 21:27
# Author     : chenshuai
# version    : python 3.9
# Description: 模型加载
"""

# BertTokenizerFast 一种分词模型
from transformers import BertModel, BertTokenizerFast, BertConfig
from transformers import logging
import config_casRel
# 取消warning日志
logging.set_verbosity_error()


def load_model(model_path: str, config_path: str, tokenizer_path: str):
    """
    加载bert模型

    :param model_path: model路径
    :param config_path: 配置文件路径
    :param tokenizer_path: 分词器路径
    :return: Bert模型
    """

    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    config = BertConfig.from_pretrained(config_path)
    bert_model = BertModel.from_pretrained(model_path, config=config)

    return tokenizer, bert_model


if __name__ == '__main__':
    print(config_casRel.bert_model_path)
    bert_tokenizer, model = load_model(config_casRel.bert_model_path, config_casRel.bert_config_path,
                                       config_casRel.tokenizer_path)

    index_list = bert_tokenizer.encode('北京欢迎我，[MASK]我想在上海工作')
    print(index_list)
    tokens = bert_tokenizer.convert_ids_to_tokens(index_list)
    print(tokens)

    print(bert_tokenizer.special_tokens_map)

    token = bert_tokenizer('北京欢迎我，[MASK]我想在上海工作', padding='max_length', return_tensors='pt', max_length=20)

    res = model(**token)

