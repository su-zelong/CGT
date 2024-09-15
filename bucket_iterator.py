import math
import random

import numpy as np
import torch


class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='text_indices', shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))   # 共有多少个batch
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(data[i * batch_size: (i + 1) * batch_size]))
        return batches

    @staticmethod
    def pad_data(batch):
        batch = list(zip(*batch))
        (text_indices_list, s_length, aspect_indices_list, aspect_mask_list,
         src_mask, selected_range_list, aa_indices_list, aa_mask_list, polarity_list) = batch
        max_lens = max(batch[1])   # max length in this batch

        af_batch = sum([len(bt) for bt in text_indices_list])

        batch_text_indices = torch.LongTensor([bt+[0]*(max_lens-len(bt)) for b in text_indices_list for bt in b])
        batch_aspect_indices = torch.LongTensor([ai+[0]*(max_lens-len(ai)) for b in aspect_indices_list for ai in b])   # !
        batch_aspect_mask = torch.LongTensor([ba+[0]*(max_lens-len(ba)) for b in aspect_mask_list for ba in b])
        batch_src_mask = torch.LongTensor([b+[0]*(max_lens-len(b)) for b in src_mask])

        selected_range_list = [j for i in selected_range_list for j in i]
        batch_selected_range = np.zeros((af_batch, 3, max_lens), dtype='int')
        for af_b in range(af_batch):
            for i in range(3):
                batch_selected_range[af_b, i][:len(selected_range_list[af_b][i])] = selected_range_list[af_b][i]
        batch_selected_range = torch.LongTensor(batch_selected_range)

        map_AS = [[idx] * len(a_i) for idx, a_i in enumerate(text_indices_list)]   # 代表第几句话
        map_AS = [m for m_list in map_AS for m in m_list]
        batch_length = torch.LongTensor([s_length[i] for i in map_AS])

        map_AS_idx = [range(len(a_i)) for a_i in text_indices_list]    # 代表这句话对应的子句aspect
        map_AS = torch.LongTensor(map_AS)
        map_AS_idx = torch.LongTensor([m for m_list in map_AS_idx for m in m_list])

        aa_indices_all = [m for m_list in aa_indices_list for m in m_list]
        aspect_num = [len(a_i) for a_i in text_indices_list]
        if len(aa_indices_all) > 0:
            aa_graph_length = torch.LongTensor([2 * num - 1 for num in aspect_num])  # 相当于length

            map_AA = [[idx] * len(m_list) for idx, m_list in enumerate(aa_indices_list)]  # [0, 0, 1, 1, 3, ...]
            map_AA = torch.LongTensor([m for m_list in map_AA for m in m_list])  # [0, 0, 1, 1, 3, ...]
            map_AA_idx = torch.LongTensor([m + len(a_i) + 1 for a_i in aa_indices_list for m in range(len(a_i))])  # 连接词标号 [2, 3, 2, 3, ...]

            aa_max_length = max([len(i) for j in aa_indices_list for i in j])
            aa_mask_all = [m for m_list in aa_mask_list for m in m_list]
            batch_aa_mask = torch.LongTensor([am+[0]*(max_lens-len(am)) for am in aa_mask_all])
            batch_aa_indices = torch.LongTensor([absa+[0]*(aa_max_length-len(absa)) for absa in aa_indices_all if len(absa) > 0])
        else:
            aa_graph_length = torch.LongTensor([])
            map_AA = torch.LongTensor([])
            map_AA_idx = torch.LongTensor([])
            batch_aa_indices = torch.LongTensor([])
            batch_aa_mask = torch.LongTensor([])

        polarity = torch.LongTensor([p for b in polarity_list for p in b])

        return (batch_text_indices, batch_length, batch_selected_range,
                batch_aspect_indices, batch_aspect_mask,
                batch_src_mask, map_AS, map_AS_idx, map_AA, map_AA_idx, batch_aa_indices, batch_aa_mask, aa_graph_length, polarity)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
