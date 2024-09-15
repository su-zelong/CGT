import json
import os.path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import get_path_and_children_dict, select_spans, text2bert_id, \
    create_aa_bert_sequence, get_long_tensor, find_inner_LCA, get_word_range


class Tokenizer(object):
    def __init__(self, word2idx=None):
        self.pos_dict = None
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def fit_on_pos(self, text):
        text = text.lower()
        pos_list = []

        import spacy
        nlp = spacy.load('en_core_web_sm')
        document = nlp(text)

        for token in document:
            pos_list.append(token.pos_)
        pos_list = list(set(pos_list))
        self.pos_dict = {pos: i for i, pos in enumerate(pos_list)}
        return len(pos_list)

    def text_to_sequence(self, sequence):
        result = []
        if type(sequence) == list:
            token = sequence
        else:
            token = sequence.split(' ')
        for w in token:
            if w in self.word2idx:
                temp = self.word2idx[w]
            else:
                temp = 1
            result.append(temp)
        if len(result) == 0:
            result = [0]
        return result

    def pos_to_sequence(self, sentence):
        sentence = sentence.lower()

        import spacy
        nlp = spacy.load('en_core_web_sm')
        document = nlp(sentence)
        pos_list = []

        for token in document:
            pos_list.append(token.pos_)

        sequence = [self.pos_dict[p] for p in pos_list]
        return sequence


def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(args, word2idx):
    embedding_matrix_file_name = './output/embed/{0}d_{1}_embedding_matrix.pkl'.format(str(args.glove_dim),
                                                                                       str(args.dataset_name))
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), args.glove_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(args.glove_dim), 1 / np.sqrt(args.glove_dim),
                                                   (1, args.glove_dim))
        word_vec = load_word_vec(args.glove_dir, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


class ABSA_Dataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSA_Dataset_Reader:
    def __init__(self, args):
        self.args = args
        print('preparing {0} dataset with GloVe'.format(args.dataset_name))
        if os.path.exists('./output/vocab/{0}_vocab.pkl'.format(args.dataset_name)):
            print('Loading {0} tokenizer...'.format(args.dataset_name))
            with open('./output/vocab/{0}_vocab.pkl'.format(args.dataset_name), 'rb') as f:
                word2idx = pickle.load(f)
                tokenizer = Tokenizer(word2idx=word2idx)
        else:
            text = ABSA_Dataset_Reader.__read_text__([args.train_dataset, args.test_dataset])
            print("No Such Vocab dir and Creating {0} tokenizer".format(args.dataset_name))
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text)
            with open('./output/vocab/{0}_vocab.pkl'.format(args.dataset_name), 'wb') as f:
                pickle.dump(tokenizer.word2idx, f)
        self.embedding_matrix = build_embedding_matrix(args, tokenizer.word2idx)
        self.train_data = ABSA_Dataset(ABSA_Dataset_Reader.__read_data__(args.train_dataset, tokenizer))
        self.test_data = ABSA_Dataset(ABSA_Dataset_Reader.__read_data__(args.test_dataset, tokenizer))

    @staticmethod
    def __read_text__(fname_list):
        text = ''
        for fname in fname_list:
            with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            for d in data:
                token = d['token']
                text_raw = ' '.join(token) + ' '
                text += text_raw
        return text

    @staticmethod
    def __read_data__(fname, tokenizer):
        result = []
        with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
        for d in data:
            token, s_length, dep_head, dep_info, con_head, con_info, aspect_list, polarity_list = d['token'], d[
                'length'], \
                d['dep_head'], d['dep_info'], d['con_head'], d['con_info'], d['aspect_list'], d['polarity']

            token = [tok.lower() for tok in token]
            con_path_dict, children_dict = get_path_and_children_dict(con_head)
            token_indices = tokenizer.text_to_sequence(token)
            mapback = [idx for idx, word in enumerate(con_info) if '[N]' not in word]
            mapback_dict = {i: j for i, j in enumerate(mapback)}

            src_mask = [1] * s_length

            text_indices_list = []
            aspect_indices_list = []
            selected_range_list = []
            aspect_mask_list = []
            for i in range(len(aspect_list)):
                aspect, a_from, a_to = aspect_list[i]['term'], aspect_list[i]['from'], aspect_list[i]['to']
                asp_from, asp_to = mapback_dict[a_from], mapback_dict[a_to - 1] + 1
                aspect_range = list(range(asp_from, asp_to))
                selected_range = select_spans(token, con_info, children_dict, mapback, con_path_dict,
                                              aspect_range)  # clause spans

                aspect_indices = tokenizer.text_to_sequence(aspect)
                aspect_mask = [0] * a_from + [1] * (a_to - a_from) + [0] * (s_length - a_to)

                text_indices_list.append(token_indices)
                aspect_indices_list.append(aspect_indices)
                selected_range_list.append(selected_range)
                aspect_mask_list.append(aspect_mask)

            choice_list = [(idx, idx + 1) for idx in range(len(aspect_list) - 1)]
            aa_indices_list = []
            aa_mask_list = []
            if choice_list:
                for a1_idx, a2_idx in choice_list:
                    a1_aspect, a2_aspect = aspect_list[a1_idx], aspect_list[a2_idx]
                    a1_asp, a1_frm, a1_to = a1_aspect['term'], a1_aspect['from'], a1_aspect['to']
                    a1_con_from, a1_con_to = mapback_dict[a1_frm], mapback[a1_to - 1] + 1
                    a1_aspect_range = list(range(a1_con_from, a1_con_to))

                    a2_asp, a2_frm, a2_to = a2_aspect['term'], a2_aspect['from'], a2_aspect['to']
                    a2_con_from, a2_con_to = mapback_dict[a2_frm], mapback[a2_to - 1] + 1
                    a2_aspect_range = list(range(a2_con_from, a2_con_to))

                    a1_lca_idx = find_inner_LCA(con_path_dict, a1_aspect_range)
                    a2_lca_idx = find_inner_LCA(con_path_dict, a2_aspect_range)

                    if a1_to <= a2_frm:
                        default_range = (a1_to, a2_frm - 1)
                    else:
                        default_range = (a2_to, a1_frm - 1)

                    a1_lca_idx, a2_lca_idx = min(a1_lca_idx, a2_lca_idx), max(a1_lca_idx, a2_lca_idx)
                    word_range = get_word_range(a1_lca_idx, a2_lca_idx, con_path_dict, children_dict, mapback,
                                                default_range)
                    select_words = token[word_range[0]: word_range[-1] + 1] if (word_range[0] <= word_range[-1]) else [
                        'and']
                    aa_indices = tokenizer.text_to_sequence(select_words)
                    if word_range[-1] < word_range[0]:
                        word_range = list(word_range)
                        word_range[1], word_range[0] = word_range[0], word_range[1]
                    aa_mask = [0] * word_range[0] + [1] * (word_range[-1] + 1 - word_range[0]) + [0] * (
                                s_length - (word_range[-1] + 1))
                    aa_mask_list.append(aa_mask)
                    aa_indices_list.append(aa_indices)
            d = (text_indices_list, s_length, aspect_indices_list, aspect_mask_list, src_mask, selected_range_list,
                 aa_indices_list, aa_mask_list, polarity_list)
            result.append(d)

        return result


class Bert_ABSA_Dataset(Dataset):
    def __init__(self, args, data_file, tokenizer):
        self.args = args
        self.ABSA_data = []

        with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)

        max_length = args.max_length
        CLS_id = tokenizer.convert_tokens_to_ids(['[CLS]'])
        SEP_id = tokenizer.convert_tokens_to_ids(['[SEP]'])

        for j in range(len(data)):
            token, s_length, dep_head, dep_info, con_head, con_info, aspect_list, polarity_list = data[j]['token'], data[j]['length'], \
                data[j]['dep_head'], data[j]['dep_info'], data[j]['con_head'], data[j]['con_info'], data[j]['aspect_list'], data[j]['polarity']

            # TOOLs
            token = [tok.lower() for tok in token]
            con_path_dict, children_dict = get_path_and_children_dict(con_head)

            # BERT
            text_raw_bert_indices, word_mapback = text2bert_id(token, tokenizer)
            text_raw_bert_indices = text_raw_bert_indices[:max_length]
            word_mapback = word_mapback[:max_length]
            mapback = [idx for idx, word in enumerate(con_info) if '[N]' not in word]
            mapback_dict = {i: j for i, j in enumerate(mapback)}

            length = word_mapback[-1] + 1
            src_mask = [1] * length
            bert_length = len(word_mapback)

            bert_sequence_list = []
            aspect_mask_list = []
            bert_aspect_mask_list = []
            selected_range_list = []

            for i in range(len(aspect_list)):
                aspect, a_from, a_to = [aspect_list[i]['term']], aspect_list[i]['from'], aspect_list[i]['to']

                asp_from, asp_to = mapback_dict[a_from], mapback[a_to - 1] + 1  # aspect 转化成mapback中的序号
                aspect_range = list(range(asp_from, asp_to))
                selected_range = select_spans(token, con_info, children_dict, mapback, con_path_dict,
                                              aspect_range)  # clause spans

                aspect_bert_ids, _ = text2bert_id(aspect, tokenizer)

                aspect_mask = [0] * a_from + [1] * (a_to - a_from) + [0] * (length - a_to)
                bert_sequence = CLS_id + text_raw_bert_indices + SEP_id + aspect_bert_ids + SEP_id
                bert_aspect_mask = [0] * (1 + bert_length + 1) + [1] * (len(aspect_bert_ids) + 1)  # bert后面的aspect

                bert_sequence_list.append(bert_sequence[:max_length + 3])
                bert_aspect_mask_list.append(bert_aspect_mask[:max_length + 3])
                selected_range_list.append(selected_range)
                aspect_mask_list.append(aspect_mask)

            # create inter aspect graph
            choice_list = [(idx, idx + 1) for idx in range(len(aspect_list) - 1)]  # 相邻两个aspect
            aa_bert_sequence_list = []  # 根据当前的aspect，生成句子中与其他aspect的连接词的Bert_sequence
            cnum = len(aspect_list) * 2 - 1
            aa_graph = np.zeros((cnum, cnum))
            if choice_list:
                cnt = 0  # 连接词
                num_aspects = len(aspect_list)
                for a1_idx, a2_idx in choice_list:

                    # ####### create inter aspect graph ########
                    if a1_idx % 2 == 0:
                        aa_graph[num_aspects + cnt][a1_idx] = 1
                        aa_graph[a2_idx][num_aspects + cnt] = 1
                    else:
                        aa_graph[a1_idx][num_aspects + cnt] = 1
                        aa_graph[num_aspects + cnt][a2_idx] = 1
                    aa_graph[a1_idx][a1_idx] = 1
                    aa_graph[a2_idx][a2_idx] = 1
                    aa_graph[num_aspects + cnt][num_aspects + cnt] = 1
                    cnt += 1

                    a1_aspect, a2_aspect = aspect_list[a1_idx], aspect_list[a2_idx]
                    aa_bert_idx = create_aa_bert_sequence(a1_aspect, a2_aspect, con_path_dict, children_dict,
                                                          mapback_dict, mapback, token, tokenizer, max_length)
                    aa_bert_sequence_list.append(CLS_id + aa_bert_idx + SEP_id)

            d = (token, length, bert_length, word_mapback, aspect_mask_list,
                 bert_sequence_list, src_mask, bert_aspect_mask_list,
                 selected_range_list, aa_graph, aa_bert_sequence_list,
                 polarity_list)

            self.ABSA_data.append(d)

    def __getitem__(self, idx):
        return self.ABSA_data[idx]

    def __len__(self):
        return len(self.ABSA_data)


def ABSA_collate_fn(batch):
    batch_size = len(batch)
    batch = list(zip(*batch))
    max_lens = max(batch[1])

    (token, length, bert_length, word_mapback, aspect_mask_list,
     bert_sequence_list, src_mask, bert_aspect_mask_List,
     selected_range_list, aa_graph, aa_bert_sequence_list,
     polarity_list) = batch

    length = torch.LongTensor(length)
    bert_length = torch.LongTensor(bert_length)
    word_mapback = get_long_tensor(word_mapback, batch_size)  # 8*43

    af_batch = sum([len(b_i) for b_i in bert_sequence_list])

    batch_bert_sequence = [bs for b in bert_sequence_list for bs in b]
    batch_bert_sequence = get_long_tensor(batch_bert_sequence, af_batch)

    batch_aspect_mask = [ba for b in aspect_mask_list for ba in b]
    batch_aspect_mask = get_long_tensor(batch_aspect_mask, af_batch)

    batch_bert_aspect_mask = [bba for b in bert_aspect_mask_List for bba in b]
    batch_bert_aspect_mask = get_long_tensor(batch_bert_aspect_mask, af_batch)

    selected_range_list = [j for i in selected_range_list for j in i]
    batch_selected_range = np.zeros((af_batch, 3, max_lens), dtype='int')
    for af_b in range(af_batch):
        for i in range(3):
            batch_selected_range[af_b, i][:len(selected_range_list[af_b][i])] = selected_range_list[af_b][i]
    batch_selected_range = torch.LongTensor(batch_selected_range)

    map_AS = [[idx] * len(a_i) for idx, a_i in enumerate(bert_sequence_list)]  # 说明每个句子下面有几个子句
    map_AS_idx = [range(len(a_i)) for a_i in bert_sequence_list]  # 每个句子下面子句标号
    map_AS = torch.LongTensor([m for m_list in map_AS for m in m_list])  # 句子本身的索引
    map_AS_idx = torch.LongTensor([m for m_list in map_AS_idx for m in m_list])

    # pad aa_graph
    aspect_num = [len(a_i) for a_i in bert_sequence_list]
    max_aa_graph = max([len(i) for i in aa_graph])
    batch_aa_graph = np.zeros((batch_size, max_aa_graph, max_aa_graph), dtype=float)
    for b in range(batch_size):
        m_len = len(aa_graph[b])
        batch_aa_graph[b, :m_len, :m_len] = aa_graph[b]
    batch_aa_graph = torch.FloatTensor(batch_aa_graph)

    # index aa_bert_sequence
    aa_bert_sequence_all = [m for m_list in aa_bert_sequence_list for m in m_list]  # 所有
    aa_batch_size = len(aa_bert_sequence_all)
    if len(aa_bert_sequence_all) > 0:
        aa_graph_length = torch.LongTensor([2 * num - 1 for num in aspect_num])  # 相当于length
        map_AA = [[idx] * len(m_list) for idx, m_list in enumerate(aa_bert_sequence_list)]  # [0, 0, 1, 1, 3, ...]
        map_AA = torch.LongTensor([m for m_list in map_AA for m in m_list])  # [0, 0, 1, 1, 3, ...]
        map_AA_idx = torch.LongTensor(
            [m + len(a_i) + 1 for a_i in aa_bert_sequence_list for m in range(len(a_i))])  # 连接词标号 [2, 3, 2, 3, ...]

        batch_aa_bert_sequence = [absa for absa in aa_bert_sequence_all if len(absa) > 0]
        batch_aa_bert_length = torch.LongTensor([len(m) - 2 for m in batch_aa_bert_sequence])
        batch_aa_bert_sequence = get_long_tensor(batch_aa_bert_sequence, aa_batch_size)
    else:
        aa_graph_length = torch.LongTensor([])
        map_AA = torch.LongTensor([])
        map_AA_idx = torch.LongTensor([])
        batch_aa_bert_sequence = torch.LongTensor([])
        batch_aa_bert_length = torch.LongTensor([])

    src_mask = get_long_tensor(src_mask, batch_size=batch_size)

    polarity = torch.LongTensor([p for b in polarity_list for p in b])

    return (length, bert_length, word_mapback, batch_aspect_mask, batch_selected_range,
            batch_bert_sequence, batch_bert_aspect_mask, src_mask, map_AS, map_AS_idx,
            map_AA, map_AA_idx, batch_aa_bert_sequence, batch_aa_bert_length, batch_aa_graph, aa_graph_length,
            polarity)
