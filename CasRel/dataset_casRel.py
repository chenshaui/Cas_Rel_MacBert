# -*- coding:utf-8 -*-

"""
# File       : dataset_casRel.py
# Time       : 2023/3/2 21:59
# Author     : chenshuai
# version    : python 3.9
# Description: 数据集
"""

import random
from load_bert_model import load_model
from data_procressing import *
import torch.utils.data as data


class dataset_cas_rel(data.Dataset):
    """
    数据集类
    """

    def __init__(self, data_type: str = 'train'):
        """
        init
        """

        data_path = train_data_path

        if data_type == 'test':
            data_path = test_data_path

        if data_type == 'dev':
            data_path = dev_data_path

        self.data = open(data_path, 'r', encoding='utf8').readlines()
        self.rel, self.rel_id = load_rel()
        self.bert_tokenizerFast, self.model = load_model(bert_model_path, bert_config_path, tokenizer_path)

    def __getitem__(self, item):
        """
        获取指定数据

        :param item: 行号
        :return: result
        """

        data = self.data[item]
        data = json.loads(data)
        # 记录的是tokenizer后的token与原来的关系,即: 每一个token具体占了几个char，中文都是一个char，英文就看单词的长度
        # tokenizer会把英文单词拆分成字母，这里需要将单词作为一个整体,所以使用BertTokenizerFast联合使用offsets_mapping
        tokenized = self.bert_tokenizerFast(data['text'], return_offsets_mapping=True)
        data['input_ids'] = tokenized['input_ids']
        data['offset_mapping'] = tokenized['offset_mapping']

        return self.parse_json(data)

    def __len__(self):
        """
        获取数据长度

        :return: 数据长度
        """

        return len(self.data)

    def parse_json(self, info: dict) -> dict:
        """
        待处理数据

        :param info: 数据
        :return: 结果字典
        """

        text = info['text']
        input_ids = info['input_ids']
        data_dict = {
            'text': text,
            'input_ids': input_ids,
            'offset_mapping': info['offset_mapping'],
            'sub_head_ids': [],
            'sub_tail_ids': [],
            'triple_list': [],  # 三元组列表
            'triple_id_list': []
        }

        # 解析三元组添加至数据字典
        for spo in info['spo_list']:
            subject_ = spo['subject']
            object_ = spo['object']['@value']
            predicate = spo['predicate']
            data_dict['triple_list'].append((subject_, predicate, object_))

            # 计算subject实体位置
            tokens = self.bert_tokenizerFast(subject_, add_special_tokens=False)
            subject_token = tokens['input_ids']
            sub_pos_id = self.get_pos_id(input_ids, subject_token)
            # 匹配失败跳过
            if not sub_pos_id:
                continue
            sub_head_id, sub_tail_id = sub_pos_id

            # 计算object实体位置
            object_tokens = self.bert_tokenizerFast(object_, add_special_tokens=False)
            obj_token = object_tokens['input_ids']
            obj_pos_id = self.get_pos_id(input_ids, obj_token)
            if not obj_pos_id:
                continue
            obj_head_id, obj_tail_id = obj_pos_id

            # 将数据添加到数据字典
            data_dict['sub_head_ids'].append(sub_head_id)
            data_dict['sub_tail_ids'].append(sub_tail_id)
            data_dict['triple_id_list'].append((
                [sub_head_id, sub_tail_id],
                self.rel_id[predicate],
                [obj_head_id, obj_tail_id]
            ))

        return data_dict

    @staticmethod
    def get_pos_id(ids, element):
        """
        获取tokenizerFast拆分的位置id

        :param ids: id列表
        :param element: 实体
        :return:
        """

        for head_id in range(len(ids)):
            tail_id = head_id + len(element)
            if ids[head_id: tail_id] == element:
                return head_id, tail_id - 1

    def collate_fn(self, batch):
        """
        批处理，dataloader的中间函数
        模型参数结构:
        [batch_mask, (batch_text: 输入文本, batch_sub_rnd: 随机取的实体), (batch_sub: 预测的实体, batch_obj_rel: 预测的关系)]

        :param batch: 批处理的数据集
        :return:
        """

        batch.sort(key=lambda x: len(x['input_ids']), reverse=True)
        # 获取最大长度
        max_len = len(batch[0]['input_ids'])
        batch_text = {
            'text': [],
            'input_ids': [],
            'offset_mapping': [],
            'triple_list': []
        }
        batch_mask = []
        batch_sub = {
            'heads_seq': [],
            'tails_seq': [],
        }
        batch_sub_rnd = {
            'head_seq': [],
            'tail_seq': [],
        }
        batch_obj_rel = {
            'heads_mx': [],
            'tails_mx': [],
        }

        for item in batch:
            input_ids = item['input_ids']
            item_len = len(input_ids)
            # 填充mask、生成mask序列
            input_ids = input_ids + [0] * (max_len - item_len)
            mask = [1] * item_len + [0] * (max_len - item_len)

            # 填充subject位置 onehot填充, 用one-hot记录首尾id位置
            sub_head_seq = one_hot_fill(max_len, item['sub_head_ids'])
            sub_tail_seq = one_hot_fill(max_len, item['sub_tail_ids'])
            # 随机选择一个subject参与训练
            if len(item['triple_id_list']) == 0:
                continue
            sub_rnd = random.choice(item['triple_id_list'])[0]
            sub_rnd_head_seq = one_hot_fill(max_len, [sub_rnd[0]])
            sub_rnd_tail_seq = one_hot_fill(max_len, [sub_rnd[1]])
            # 随机subject计算relation矩阵
            obj_head_mx = [[0] * RELATION_SIZE for i in range(max_len)]
            obj_tail_mx = [[0] * RELATION_SIZE for i in range(max_len)]
            for triple in item['triple_id_list']:
                rel_id = triple[1]
                head_id, tail_id = triple[2]
                if triple[0] == sub_rnd:
                    obj_head_mx[head_id][rel_id] = 1
                    obj_tail_mx[tail_id][rel_id] = 1

            # 填充数据 (数据必须在这里统一填充，因为方法存在continue关键字)
            batch_text['text'].append(item['text'])
            batch_text['input_ids'].append(input_ids)
            batch_text['offset_mapping'].append(item['offset_mapping'])
            batch_text['triple_list'].append(item['triple_list'])
            batch_sub['heads_seq'].append(sub_head_seq)
            batch_sub['tails_seq'].append(sub_tail_seq)
            batch_sub_rnd['head_seq'].append(sub_rnd_head_seq)
            batch_sub_rnd['tail_seq'].append(sub_rnd_tail_seq)
            batch_obj_rel['heads_mx'].append(obj_head_mx)
            batch_obj_rel['tails_mx'].append(obj_tail_mx)
            # 添加至batch_mask
            batch_mask.append(mask)

        return batch_mask, (batch_text, batch_sub_rnd), (batch_sub, batch_obj_rel)


def one_hot_fill(length, pos):
    """
    one hot 填充

    :param length: 语句长度
    :param pos: 待填充位置
    :return: 填充后的位置
    """

    return [1 if hot_pos in pos else 0 for hot_pos in range(length)]


def get_triple_list(sub_head_ids, sub_tail_ids, model, encoded_text, text, mask, offset_mapping):
    id2rel, _ = get_rel()
    triple_list = []
    for sub_head_id in sub_head_ids:
        sub_tail_ids = sub_tail_ids[sub_tail_ids >= sub_head_id]
        if len(sub_tail_ids) == 0:
            continue
        sub_tail_id = sub_tail_ids[0]
        if mask[sub_head_id] == 0 or mask[sub_tail_id] == 0:
            continue
        # 根据位置信息反推出 subject 文本内容
        sub_head_pos_id = offset_mapping[sub_head_id][0]
        sub_tail_pos_id = offset_mapping[sub_tail_id][1]
        subject_text = text[sub_head_pos_id:sub_tail_pos_id]

        # 根据 subject 计算出对应 object 和 relation
        sub_head_seq = torch.tensor(one_hot_fill(len(mask), sub_head_id)).to(DEVICE)
        sub_tail_seq = torch.tensor(one_hot_fill(len(mask), sub_tail_id)).to(DEVICE)

        pred_obj_head, pred_obj_tail = model.get_obj_for_specific_sub( \
            encoded_text.unsqueeze(0), sub_head_seq.unsqueeze(0), sub_tail_seq.unsqueeze(0))

        # 按分类找对应关系
        pred_obj_head = pred_obj_head[0].T
        pred_obj_tail = pred_obj_tail[0].T
        for j in range(len(pred_obj_head)):
            obj_head_ids = torch.where(pred_obj_head[j] > OBJ_HEAD_BAR)[0]
            obj_tail_ids = torch.where(pred_obj_tail[j] > OBJ_TAIL_BAR)[0]
            for obj_head_id in obj_head_ids:
                obj_tail_ids = obj_tail_ids[obj_tail_ids >= obj_head_id]
                if len(obj_tail_ids) == 0:
                    continue
                obj_tail_id = obj_tail_ids[0]
                if mask[obj_head_id] == 0 or mask[obj_tail_id] == 0:
                    continue
                # 根据位置信息反推出 object 文本内容，mapping中已经有移位，不需要再加1
                obj_head_pos_id = offset_mapping[obj_head_id][0]
                obj_tail_pos_id = offset_mapping[obj_tail_id][1]
                object_text = text[obj_head_pos_id:obj_tail_pos_id]
                triple_list.append((subject_text, id2rel[j], object_text))
    return list(set(triple_list))


def get_rel():
    """
    获取真实值
    """

    df = pd.read_csv(REL_PATH, names=['rel', 'id'])
    return df['rel'].tolist(), dict(df.values)


def report(model, encoded_text, pred_y, batch_text, batch_mask):
    # 计算三元结构，和统计指标
    pred_sub_head, pred_sub_tail, _, _ = pred_y
    true_triple_list = batch_text['triple_list']
    pred_triple_list = []

    correct_num, predict_num, gold_num = 0, 0, 0

    # 遍历batch
    for i in range(len(pred_sub_head)):
        text = batch_text['text'][i]

        true_triple_item = true_triple_list[i]
        mask = batch_mask[i]
        offset_mapping = batch_text['offset_mapping'][i]

        sub_head_ids = torch.where(pred_sub_head[i] > SUB_HEAD_BAR)[0]
        sub_tail_ids = torch.where(pred_sub_tail[i] > SUB_TAIL_BAR)[0]

        pred_triple_item = get_triple_list(sub_head_ids, sub_tail_ids, model, \
                                           encoded_text[i], text, mask, offset_mapping)

        # 统计个数
        correct_num += len(set(true_triple_item) & set(pred_triple_item))
        predict_num += len(set(pred_triple_item))
        gold_num += len(set(true_triple_item))

        pred_triple_list.append(pred_triple_item)

    precision = correct_num / (predict_num + EPS)
    recall = correct_num / (gold_num + EPS)
    f1_score = 2 * precision * recall / (precision + recall + EPS)
    print('\t correct_num:', correct_num, 'predict_num:', predict_num, 'gold_num:', gold_num)
    print('\t precision:%.3f' % precision, 'recall:%.3f' % recall, 'f1_score:%.3f' % f1_score)

# if __name__ == '__main__':
#     dataset = dataset_cas_rel()
#     loader = data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn)
# print(iter(loader).next())
