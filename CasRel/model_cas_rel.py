# -*- coding:utf-8 -*-

"""
# File       : model_cas_rel.py
# Time       : 2023/3/4 12:20
# Author     : chenshuai
# version    : python 3.9
# Description: 
"""

import torch.optim
import torch.nn.functional as F
import torch.nn as nn
from dataset_casRel import *


class Cas_Rel_Model(nn.Module):
    """
    模型类
    """

    def __init__(self):
        super().__init__()
        _, self.bert_model = load_model(bert_model_path, bert_config_path, tokenizer_path)
        # bert不参与训练
        # fixme 也可以参与训练，可测试一下效果
        for name, param in self.bert_model.named_parameters():
            param.requires_grad = False
        # 定义各个层的线层
        self.sub_head_linear = nn.Linear(BERT_SIZE, 1)
        self.sub_tail_linear = nn.Linear(BERT_SIZE, 1)
        self.obj_head_linear = nn.Linear(BERT_SIZE, RELATION_SIZE)
        self.obj_tail_linear = nn.Linear(BERT_SIZE, RELATION_SIZE)

    def forward(self, inputs, mask):
        """
        forward

        :param inputs: 输入数据
        :param mask: mask矩阵
        :return:
        """

        input_ids, sub_head_seq, sub_tail_seq = inputs
        encode_text = self.get_encode_text(input_ids, mask)
        # 预测subject首尾序列
        pre_sub_head, pre_sub_tail = self.get_encode_subs(encode_text)
        # 预测relation-object矩阵
        pre_obj_head, pre_obj_tail = self.get_obj_for_specific_sub(encode_text, sub_head_seq, sub_tail_seq)

        return encode_text, (pre_sub_head, pre_sub_tail, pre_obj_head, pre_obj_tail)

    def loss_fn(self, real_y, pre_y, mask):
        """
        损失函数

        :param real_y: 真实值
        :param pre_y: 预测值
        :param mask: mask矩阵
        :return:
        """

        def calc_loss(pred, true, mask):
            true = true.float()
            # pred.shape (b, c, 1) -> (b, c)
            pred = pred.squeeze(-1)
            weight = torch.where(true > 0, CLS_WEIGHT_COEF[1], CLS_WEIGHT_COEF[0])
            loss = F.binary_cross_entropy(pred, true, weight=weight, reduction='none')
            if loss.shape != mask.shape:
                mask = mask.unsqueeze(-1)
            return torch.sum(loss * mask) / torch.sum(mask)

        pred_sub_head, pred_sub_tail, pred_obj_head, pred_obj_tail = pre_y
        true_sub_head, true_sub_tail, true_obj_head, true_obj_tail = real_y
        return calc_loss(pred_sub_head, true_sub_head, mask) * SUB_WEIGHT_COEF + \
               calc_loss(pred_sub_tail, true_sub_tail, mask) * SUB_WEIGHT_COEF + \
               calc_loss(pred_obj_head, true_obj_head, mask) + \
               calc_loss(pred_obj_tail, true_obj_tail, mask)

    def get_obj_for_specific_sub(self, encode_text, sub_head_seq, sub_tail_seq):
        """
        预测relation-object矩阵

        :param encode_text: 编码后的词向量 (x, y, 768)
        :param sub_head_seq: subject的头序列 (x, y)
        :param sub_tail_seq: subject的尾序列 (x, y)
        :return: 预测的obj首尾矩阵 (x, y, rel_size)
        """

        # (x, y) -> (x, 1, y)
        sub_head_seq = sub_head_seq.unsqueeze(1).float()
        sub_tail_seq = sub_tail_seq.unsqueeze(1).float()
        # 即结构图中实体序列与词向量拼接的部分, 先对头尾进行处理 (x, 1, 768)
        sub_head = torch.matmul(sub_head_seq, encode_text)
        sub_tail = torch.matmul(sub_tail_seq, encode_text)
        # 实体词向量相加不加整体和，而是首尾相加除以2 encode_text (x, y, 768) 得到句子新的编码，
        encode_text = encode_text + (sub_head + sub_tail) / 2
        # (x, y, rel_size)
        pre_obj_head = torch.sigmoid(self.obj_head_linear(encode_text))
        pre_obj_tail = torch.sigmoid(self.obj_tail_linear(encode_text))

        return pre_obj_head, pre_obj_tail

    def get_encode_subs(self, encoded_text):
        """
        从encode文本中提取subject首尾

        :param encoded_text: 编码后的词向量
        :return: 分类结果
        """

        pre_sub_head = torch.sigmoid(self.sub_head_linear(encoded_text))
        pre_sub_tail = torch.sigmoid(self.sub_tail_linear(encoded_text))

        return pre_sub_head, pre_sub_tail

    def get_encode_text(self, input_ids, mask):
        """
        获取encode结果

        :param input_ids: 输入数据id列表
        :param mask: mask 矩阵
        :return: encode结果 （只要词向量）
        """

        return self.bert_model(input_ids, attention_mask=mask)[0]


# if __name__ == '__main__':
#     model = Cas_Rel_Model().to(DEVICE)
#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#
#     dataset = dataset_cas_rel()
#     for epo in range(EPOCH):
#         loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_fn)
#         for index, (batch_mask, batch_x, batch_y) in enumerate(loader):
#             batch_text, batch_sub_rnd = batch_x
#             batch_sub, batch_obj_rel = batch_y
#
#             input_mask = torch.tensor(batch_mask).to(DEVICE)
#             inputs = (
#                 torch.tensor(batch_text['input_ids']).to(DEVICE),
#                 torch.tensor(batch_sub_rnd['head_seq']).to(DEVICE),
#                 torch.tensor(batch_sub_rnd['tail_seq']).to(DEVICE)
#             )
#             model(inputs, input_mask)
#             exit()
