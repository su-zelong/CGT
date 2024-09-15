from copy import deepcopy

import torch


def get_child_list(mapback_dict, children_dict):
    """
    :param mapback_dict: {3: 0, 4: 1, 7: 2, 10: 3, 12: 4, 13: 5, 14: 6, 15: 7}
    :param children_dict: {0: [1], 1: [2, 5, 15], 2: [3, 4], 5: [6, 8], 6: [7], 8: [9], 9: [10, 11], 11: [12, 13, 14]}
    :return: children_dict with all leaf
    """
    result = deepcopy(children_dict)
    for node in sorted(children_dict.keys(), reverse=True):
        child_list = children_dict[node]
        leaf_list = []
        for n in child_list:
            temp = []
            if n in mapback_dict.keys():
                temp.append(n)
            else:
                next_step = result[n]
                for i in next_step:
                    temp.append(i)
            for i in temp:
                leaf_list.append(i)
        result[node] = leaf_list

    re_result = deepcopy(result)
    for node, child in result.items():
        temp = []
        for n in child:
            temp.append(mapback_dict[n])
            re_result[node] = temp

    return re_result


def get_path_and_children_dict(heads):
    path_dict = {}
    remain_nodes = list(range(len(heads)))
    delete_nodes = []

    while len(remain_nodes) > 0:
        for idx in remain_nodes:
            # 初始状态
            if idx not in path_dict:
                path_dict[idx] = [heads[idx]]  # no self
                if heads[idx] == -1:
                    delete_nodes.append(idx)  # need delete root
            else:
                last_node = path_dict[idx][-1]
                if last_node not in remain_nodes:
                    path_dict[idx].extend(path_dict[last_node])
                    delete_nodes.append(idx)
                else:
                    path_dict[idx].append(heads[last_node])
        # remove nodes
        for del_node in delete_nodes:
            remain_nodes.remove(del_node)
        delete_nodes = []

    # 这一部分只保留token中的分词
    children_dict = {}
    for x, l in path_dict.items():
        if l[0] == -1:
            continue
        if l[0] not in children_dict:
            children_dict[l[0]] = [x]
        else:
            children_dict[l[0]].append(x)

    return path_dict, children_dict


def form_layers_range(path_dict, con_info):
    sorted_path_dict = sorted(path_dict.items(), key=lambda x: len(x[1]))
    layers = {}
    node2layerid = {}
    for cid, path_dict in sorted_path_dict[::-1]:    # -1: 从叶子到根

        length = len(path_dict) - 1
        if length not in layers:
            layers[length] = [cid]      # {‘层’: 'node1, node2, ... ...'}
            node2layerid[cid] = length      # {'node1': layer}
        else:
            layers[length].append(cid)
            node2layerid[cid] = length

    layers = sorted(layers.items(), key=lambda x: x[0])
    layers = [(cid, sorted(l)) for cid, l in layers]  # or [(cid,l.sort()) for cid,l in layers]

    # 筛选出从句部分, 但是有可能导致 new_layer=0
    new_layers = []
    l_t = 0
    for layer in layers:
        l, node_list = layer[0], layer[1]
        node_tags = [con_info[i] for i in node_list]
        if 'S[N]' in node_tags:
            temp = (l_t, node_list)
            new_layers.append(temp)
            l_t += 1

    return new_layers, node2layerid


def find_inner_LCA(path_dict, aspect_range):
    path_range = [[x] + path_dict[x] for x in aspect_range]   # aspect_path = [aspect_path + aspect]
    path_range.sort(key=lambda l: len(l))

    LCA_node = None
    for idx in range(len(path_range[0])):
        flag = True
        for pid in range(1, len(path_range)):
            if path_range[0][idx] not in path_range[pid]:
                flag = False  # 其中一个不在
                break

        if flag:  # 都在
            LCA_node = path_range[0][idx]
            break  # already find
    return LCA_node


def select_related_layers(length, con_lca, con_path_dict, con_info, children_leaf_dict):
    select_tag = ['TOP', 'S']
    path = con_path_dict[con_lca]   # aspect path

    selected_range_list = []
    span_mask = []
    for node in path:
        if con_info[node][:-3] in select_tag[:-1]:
            # this node is target node
            influence_range = children_leaf_dict[node]   # 找到这个节点对应的能影响真实节点的范围，这个children理应是连续的
            selected_range_list.append(influence_range)
            temp = [0] * length
            for i in influence_range:
                temp[i] = 1
            span_mask.append(temp)

    return span_mask


def select_spans(token, con_info, children_dict, mapback, con_path_dict, aspect_range):
    """
    为矩阵当中每一层生成相应的邻接矩阵
    :param token: ['great', 'laptop', ...]
    :param con_info: 当前句子的成分依存关系列表 ['Top', 'S', 'NP', 'VP', ...]
    :param children_dict: {0: [1], 1: [2, 5, 15], 2: [3, 4], 5: [6, 8], 6: [7], 8: [9], 9: [10, 11], 11: [12, 13, 14]}
    :param mapback:
    :param con_path_dict: {0: [-1], 1: [0, -1], 2: [1, 0, -1], 3: [2, 1, 0, -1], 4: [2, 1, 0, -1], ... ...}
    :param aspect_range: [list]
    :return: 返回每一层的依赖关系邻接矩阵
    """
    length = len(token)
    con_lca = find_inner_LCA(con_path_dict, aspect_range)   # 最小共同祖先
    mapback_dict = {j: i for i, j in enumerate(mapback)}
    children_leaf_dict = get_child_list(mapback_dict, children_dict)    # 每个节点对应的影响真是节点的范围
    span_mask = select_related_layers(length, con_lca, con_path_dict, con_info, children_leaf_dict)    # 选择与aspect_con_lca有关的上层layers

    # select 3 layers
    if len(span_mask) <= 3:
        range_pad = span_mask[-1] if len(span_mask) > 0 else [1] * length
        selected_range_list = span_mask + [range_pad] * (3 - len(span_mask))   # 不够填1
    else:
        gap = len(span_mask) // (3 - 1)
        selected_range_list = [span_mask[gap * i] for i in range(3 - 1)] + [span_mask[-1]]  # 之前的选两层，但是-1这一层一定要有

    return selected_range_list


def get_word_range(lca_A, lca_B, path_dict, children, mapback, default_range):
    LCA, pathA, pathB = find_LCA_and_PATH([lca_A] + path_dict[lca_A], [lca_B] + path_dict[lca_B])
    inner_node_LCA = children[LCA][children[LCA].index(pathA[-1]) + 1:children[LCA].index(pathB[-1])] if (
            len(pathA) and len(pathB)) else []
    word_range = FindS(inner_node_LCA, children, mapback) if len(inner_node_LCA) > 0 else default_range
    return word_range


def find_LCA_and_PATH(A, B):
    for idx in range(min(len(A), len(B))):
        if A[idx] in B:
            return A[idx], A[:idx], B[:B.index(A[idx])]
        elif B[idx] in A:
            return B[idx], A[:A.index(B[idx])], B[:idx]
    return -1, A[:-1], B[:-1]


def FindS(l, children, mapback):
    def inner_Find(x, index):
        if x[index] not in children:
            return x[index]
        else:
            return inner_Find(children[x[index]], index)

    return mapback.index(inner_Find(l, 0)), mapback.index(inner_Find(l, -1))

def get_long_tensor(tokens_list, batch_size):
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s)] = torch.LongTensor(s)
    return tokens


def text2bert_id(token, tokenizer):
    re_token = []
    word_mapback = []
    for idx, word in enumerate(token):
        temp = tokenizer.tokenize(word)  # bert_token
        re_token.extend(temp)
        word_mapback.extend([idx] * len(temp))  # 一个词分成了几个
    re_id = tokenizer.convert_tokens_to_ids(re_token)
    return re_id, word_mapback


def create_aa_bert_sequence(a1_aspect, a2_aspect, con_path_dict, children_dict, mapback_dict, mapback, token, tokenizer, max_length):
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
    word_range = get_word_range(a1_lca_idx, a2_lca_idx, con_path_dict, children_dict, mapback, default_range)
    select_words = token[word_range[0]: word_range[-1] + 1] if (word_range[0] <= word_range[-1]) else ['and']

    aa_bert_ids, _ = text2bert_id(select_words, tokenizer)
    aa_bert_idx = aa_bert_ids[:max_length]

    return aa_bert_idx
