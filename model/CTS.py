import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.autograd import Variable

from model_utils import LayerNorm, CosAttention, sequence_mask, AA_Attn, average_adj, select, AA_GCN


class Clause_encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.in_dim = args.glove_dim
        self.layer_norm = LayerNorm(self.in_dim)
        self.a_a = [CosAttention(args).to(args.device) for _ in range(args.aa_heads)]
        self.sem_combine = nn.Linear(self.in_dim * args.graph_num, self.in_dim)

    def forward(self, input_embed, batch_selected_range, batch_aspect_mask, batch_src_mask):
        clause_embed = batch_selected_range.unsqueeze(-1) * input_embed.unsqueeze(1)
        clause_embed = self.layer_norm(clause_embed)

        asp_wn = batch_aspect_mask.sum(dim=1).unsqueeze(-1)

        con_output = None
        aspect_embed = (input_embed * batch_aspect_mask.unsqueeze(-1).repeat(1, 1, self.in_dim)).sum(dim=1) / asp_wn
        for s in range(self.args.graph_num):
            c_q = clause_embed[:, s, :, :]
            c_out = [a(c_q, aspect_embed, batch_src_mask).unsqueeze(1) for a in self.a_a]  # b*d n_head=1
            c_output = torch.cat(c_out, dim=1)
            c_output = c_output.mean(dim=1)  # b*d
            con_output = torch.cat([con_output, c_output], dim=-1) if con_output is not None else c_output
        clause_output = F.relu(self.sem_combine(con_output))
        clause_output = (clause_output * batch_aspect_mask.unsqueeze(-1).repeat(1, 1, self.in_dim)).sum(dim=1) / asp_wn

        return clause_output

class AA_encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.AA_Attn = AA_Attn(args.AA_heads, args.glove_dim, dropout=0.1)
        self.gcn_weight = nn.Linear(args.glove_dim, args.glove_dim)

    def forward(self, input_embed, clause_output, batch_aa_mask, aa_graph_length, map_AA, map_AA_idx, map_AS, map_AS_idx):
        input_embed = input_embed[map_AA]
        aa_embed = (input_embed * batch_aa_mask.unsqueeze(-1)).sum(dim=1)

        B = map_AS.max() + 1
        L = map_AA_idx.max() + 1

        inner_embed = torch.zeros((B, L, clause_output.shape[-1]), device=self.args.device)
        inner_embed[map_AA, map_AA_idx] = aa_embed
        inner_embed[map_AS, map_AS_idx] = clause_output

        aa_graph_key_padding_mask = sequence_mask(aa_graph_length)
        mask_indi = (~aa_graph_key_padding_mask).int()
        AA_attn = self.AA_Attn(inner_embed, inner_embed, mask_indi.unsqueeze(1))
        AA_attn = average_adj(self.args.AA_heads, AA_attn, mask_indi)
        AA_attn = select(AA_attn, self.args.top_k) * AA_attn
        AA_output = AA_GCN(inner_embed, AA_attn, self.gcn_weight)

        need_change = (aa_graph_length[map_AS] > 1)
        AA_output = AA_output[map_AS, map_AS_idx]
        AA_features = AA_output * need_change.unsqueeze(dim=-1) + clause_output * ~need_change.unsqueeze(dim=-1)

        return AA_features

class CTS(nn.Module):
    def __init__(self, args, embedding_matrix):
        super().__init__()
        self.args = args
        self.embed_dim = args.glove_dim
        self.word_embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.embed_dropout = nn.Dropout(0.3)

        self.lstm = nn.LSTM(self.embed_dim, self.args.lstm_hidden_dim, self.args.lstm_layers, batch_first=True, bidirectional=True)  # out_dim=300
        self.lstm_dropout = nn.Dropout(0.1)
        self.layer_norm = LayerNorm(args.glove_dim)

        self.Clause_encoder = Clause_encoder(self.args)
        self.AA_encoder = AA_encoder(self.args)

        self.classifier = nn.Linear(args.glove_dim, args.num_classes)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(batch_size, self.args.lstm_hidden_dim, self.args.lstm_layers, self.args.bi_direct)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True, enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.lstm(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)  # 16*35*100
        return rnn_outputs

    def forward(self, inputs):
        batch_text_indices, batch_length, batch_selected_range, batch_aspect_indices, batch_aspect_mask,\
            batch_src_mask, map_AS, map_AS_idx, map_AA, map_AA_idx, batch_aa_indices, batch_aa_mask, aa_graph_length, polarity = inputs

        text_embed = self.word_embed(batch_text_indices)
        text_embed = self.embed_dropout(text_embed)

        self.lstm.flatten_parameters()
        input_embed = self.lstm_dropout(self.encode_with_rnn(rnn_inputs=text_embed, seq_lens=np.array(batch_length.cpu(), dtype=int),
                                                             batch_size=text_embed.size()[0]))

        # clause embed
        clause_output = self.Clause_encoder(input_embed, batch_selected_range, batch_aspect_mask, batch_src_mask[map_AS])

        # inter aspect
        if self.args.Inter_aspect and map_AA.numel():
            aa_output = self.AA_encoder(input_embed, clause_output, batch_aa_mask, aa_graph_length, map_AA, map_AA_idx, map_AS, map_AS_idx)
        else:
            aa_output = None

        final_output = aa_output + clause_output if aa_output is not None else clause_output
        result = self.classifier(final_output)

        return result


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)  # (2, 6, 50)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)  # 2*16*50
    return h0.cuda(), c0.cuda()
