import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from model_utils import LayerNorm, sequence_mask, CosAttention, AA_Attn, average_adj, select, \
    AA_GCN


class Clause_encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.layer_norm = LayerNorm(args.hidden_dim)
        self.bert_drop = nn.Dropout(args.bert_dropout)
        self.a_a = [CosAttention(args).to(args.device) for _ in range(args.aa_heads)]
        self.sem_combine = nn.Linear(args.hidden_dim * args.graph_num, args.hidden_dim)

    def forward(self, bert_out, batch_span_list, aspect_mask, src_mask):
        clause_out = batch_span_list.unsqueeze(-1) * bert_out.unsqueeze(1)
        clause_embed = self.layer_norm(clause_out)  # layer_norm
        clause_embed = self.bert_drop(clause_embed)

        # bert_out = self.layer_norm(bert_out)
        bert_out = self.bert_drop(bert_out)

        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)  # aspect

        con_output = None
        aspect_embed = (bert_out * aspect_mask.unsqueeze(-1).repeat(1, 1, self.args.hidden_dim)).sum(dim=1) / asp_wn  # b*l*d.sum(dim=1)/len(asp)
        for s in range(self.args.graph_num):
            c_q = clause_embed[:, s, :, :]

            #   ######### Aspect focus Attn ########
            c_out = [a(c_q, aspect_embed, src_mask).unsqueeze(1) for a in self.a_a]  # b*d n_head=1
            c_output = torch.cat(c_out, dim=1)
            c_output = c_output.mean(dim=1)  # b*d
            con_output = torch.cat([con_output, c_output], dim=-1) if con_output is not None else c_output
        # for s in range(self.args.graph_num):
        #     c_q = clause_embed[:, s, :, :]
        #
        #     con_output = torch.cat([con_output, c_q], dim=-1) if con_output is not None else c_q

        # fully connected
        clause_output = F.relu(self.sem_combine(con_output))
        # asp mask
        clause_output = (clause_output * aspect_mask.unsqueeze(-1).repeat(1, 1, self.args.hidden_dim)).sum(dim=1) / asp_wn

        return clause_output


class AA_encoder(nn.Module):
    def __init__(self, args, layer_norm_eps=1e-5):
        super().__init__()
        self.args = args
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        # self.bert_dropout = nn.Dropout(0.0)
        self.dense = nn.Linear(args.bert_dim, args.hidden_dim)

        self.AA_Attn = AA_Attn(args.AA_heads, args.bert_dim + args.hidden_dim, dropout=0.1)

        self.gcn_weight = nn.Linear(args.bert_dim + args.hidden_dim, args.bert_dim + args.hidden_dim)

    def forward(self, batch_aa_bert_sequence, batch_aa_bert_length, aa_graph_length, clause_output,
                map_AA, map_AA_idx, map_AS, map_AS_idx):
        inner_bert_output = self.bert_model(batch_aa_bert_sequence)
        inner_bert_out, inner_pooled_out = inner_bert_output.last_hidden_state, inner_bert_output.pooler_output
        bert_seq_indi = ~sequence_mask(batch_aa_bert_length).unsqueeze(dim=-1)
        inner_bert_out = inner_bert_out[:, 1:max(batch_aa_bert_length) + 1, :] * bert_seq_indi.float()
        inner_bert_out = self.dense(inner_bert_out)
        inner_input = torch.cat((inner_bert_out.sum(dim=1), inner_pooled_out), dim=-1)

        B = map_AS.max() + 1  # inner batch 真实的batch
        L = map_AA_idx.max() + 1  # 中间句子长度
        inner_embed = torch.zeros((B, L, clause_output.shape[-1]),
                                  device=self.args.device)  # embed([food, env, service, and ,but])

        inner_embed[map_AA, map_AA_idx] = inner_input  # 连词
        inner_embed[map_AS, map_AS_idx] = clause_output  # aspect 这样组成了 [food, env, service, and, but]的向量 这里要不要换成GCN试试 用attention做邻接矩阵

        # create mask
        aa_graph_key_padding_mask = sequence_mask(aa_graph_length)

        mask_indi = (~aa_graph_key_padding_mask).int()
        AA_attn = self.AA_Attn(inner_embed, inner_embed, mask_indi.unsqueeze(1))
        AA_attn = average_adj(self.args.AA_heads, AA_attn, mask_indi)
        AA_attn = select(AA_attn, self.args.top_k) * AA_attn

        AA_output = inner_embed
        AA_output = AA_GCN(AA_output, AA_attn, self.gcn_weight)

        need_change = (aa_graph_length[map_AS] > 1)
        AA_output = AA_output[map_AS, map_AS_idx]
        AA_features = AA_output * need_change.unsqueeze(dim=-1) + clause_output * ~need_change.unsqueeze(dim=-1)

        return AA_features


class CTS_BERT(nn.Module):
    def __init__(self, args, bert):
        super().__init__()
        self.args = args
        self.bert = bert
        self.bert_out_dim = args.bert_dim
        self.mem_dim = args.bert_dim // 2
        self.dense = nn.Linear(args.bert_dim, args.hidden_dim)

        self.Clause_encoder = Clause_encoder(self.args)

        self.AA_encoder = AA_encoder(self.args)

        self.classifier = nn.Linear(args.bert_dim + args.hidden_dim, args.num_classes)
        # self.classifier = nn.Linear(args.hidden_dim + self.bert_out_dim, args.num_classes)

    def average_bert(self, sequence_output, bert_length, word_mapback):
        bert_seq_indi = ~sequence_mask(bert_length).unsqueeze(dim=-1)
        bert_out = sequence_output[:, 1:max(bert_length) + 1, :]
        word_mapback_one_hot = (F.one_hot(word_mapback).float() * bert_seq_indi.float()).transpose(1, 2)
        bert_out = torch.bmm(word_mapback_one_hot.float(), self.dense(bert_out))
        wnt = word_mapback_one_hot.sum(dim=-1)
        wnt.masked_fill_(wnt == 0, 1)
        bert_out = bert_out / wnt.unsqueeze(dim=-1)
        return bert_out

    def forward(self, inputs):
        length, bert_length, word_mapback, batch_aspect_mask, batch_selected_range, batch_bert_sequence, \
            batch_bert_aspect_mask, src_mask, map_AS, map_AS_idx, map_AA, map_AA_idx, batch_aa_bert_sequence, \
            batch_aa_bert_length, batch_aa_graph, aa_graph_length, polarity = inputs

        bert_output = self.bert(batch_bert_sequence, token_type_ids=batch_bert_aspect_mask)
        sequence_output, pooled_output = bert_output.last_hidden_state, bert_output.pooler_output

        bert_out = self.average_bert(sequence_output, bert_length[map_AS], word_mapback[map_AS])

        # ######### clause embed ##########
        clause_output = self.Clause_encoder(bert_out, batch_selected_range, batch_aspect_mask, src_mask[map_AS])
        clause_output = torch.cat((clause_output, pooled_output), dim=-1)  # 968

        # ######### enhance inter aspect ########
        if self.args.Inter_aspect and map_AA.numel():
            aa_output = self.AA_encoder(batch_aa_bert_sequence, batch_aa_bert_length, aa_graph_length,
                                        clause_output, map_AA, map_AA_idx, map_AS, map_AS_idx)
        else:
            aa_output = None

        final_output = aa_output + clause_output if aa_output is not None else clause_output
        #
        # cap_out = self.capsule(final_output)
        # cap_out = cap_out.view(final_output.size(0), -1)

        result = self.classifier(final_output)

        return result
