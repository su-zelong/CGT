import argparse

import torch
import numpy as np
import random

from Instruct import Instructor
from model.CTS import CTS
from model.CTS_BERT import CTS_BERT


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='CTS_BERT')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default=None, help='cpu or gpu')
    parser.add_argument('--cuda', default='0', type=str)
    parser.add_argument('--max_length', default=200)

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_epoch', default=20, type=int, help='each repeat training epoch')
    parser.add_argument('--l2reg', default=1e-5, type=float, help='Normalization of l2')
    parser.add_argument('--num_classes', default=3, type=int, help='positive, negative or neural')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--con_mark', default='[N]', type=str)
    parser.add_argument('--graph_num', default=3, type=int)

    parser.add_argument('--dataset_name', default='res14', type=str)

    # Ablation param
    parser.add_argument('--con_layers', default=1, type=int)
    parser.add_argument('--top_k', default=3, type=int)

    # model param
    parser.add_argument('--aa_heads', default=1, type=int)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--Inter_aspect', default=True, type=bool)
    parser.add_argument('--AA_heads', default=2, type=int, help='Inter aspect Attention heads')
    parser.add_argument('--embed_dim', default=968, type=int, help='AA_Attention input embed dim')
    parser.add_argument('--dropout', default=0.1, type=float)

    parser.add_argument('--ka_layers', default=1, type=int)

    # GloVe training
    parser.add_argument('--glove_dir', default=r'D:\Academic\Glove\glove.840B.300d.txt', type=str, help='Glove based on standford dir')
    parser.add_argument('--glove_dim', default=300, type=int)
    parser.add_argument('--glove_learning_rate', default=0.001, type=float)

    # LSTM
    parser.add_argument('--bi_direct', default=True, type=bool, help='whether use bi-direct lstm')
    parser.add_argument('--lstm_input_dim', default=300, type=int, help='input dimension of lstm')
    parser.add_argument('--lstm_hidden_dim', default=150, type=int, help='hidden dimension of lstm')
    parser.add_argument('--lstm_output_dim', default=300, type=int, help='output dim of lstm if bi')
    parser.add_argument('--lstm_dropout', default=0.1, type=float)
    parser.add_argument('--lstm_layers', default=1, type=int, help='lstm layers')

    # Bert training
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay if we apply some.')
    parser.add_argument('--pretrained_bert_name', default='bert_base_uncased')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Epsilon for Adam optimizer.')
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--bert_dropout', type=float, default=0.1, help='BERT dropout rate.')
    parser.add_argument('--bert_lr', default=2e-5, type=float)

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    setup_seed(args.seed)

    data_file = {               # original data
        'lap14': {
            'train': './datasets/lap14/laptop_train_new.json',
            'test': './datasets/lap14/laptop_test_new.json'
        },
        'res14': {
            'train': './datasets/res14/restaurant_train_new.json',
            'test': './datasets/res14/restaurant_test_new.json'
        },
        'res15': {
            'train': './datasets/res15/restaurant_train_new.json',
            'test': './datasets/res15/restaurant_test_new.json'
        },
        'res16': {
            'train': './datasets/res16/restaurant_train_new.json',
            'test': './datasets/res16/restaurant_test_new.json'
        },
        'twitter': {
            'train': './datasets/twitter/twitter_train_new.json',
            'test': './datasets/twitter/twitter_test_new.json'
        }
    }
    ARTS_file = {
        'lap14': {
            'entire': './datasets/ARTS/laptop_test_ARTS.json',
            'adv_1': './datasets/ARTS/laptop_test_ARTS_adv1.json',
            'adv_2': './datasets/ARTS/laptop_test_ARTS_adv2.json',
            'adv_3': './datasets/ARTS/laptop_test_ARTS_adv3.json'
        },
        'res14': {
            'entire': './datasets/ARTS/rest_test_ARTS.json',
            'adv_1': './datasets/ARTS/rest_test_ARTS_adv1.json',
            'adv_2': './datasets/ARTS/rest_test_ARTS_adv2.json',
            'adv_3': './datasets/ARTS/rest_test_ARTS_adv3.json'
        }
    }
    model_file = {
        'CTS': CTS,
        'CTS_BERT': CTS_BERT,
    }

    args.model = model_file[args.model_name]
    args.train_dataset = data_file[args.dataset_name]['train']    # a dict with train_file and test_file

    args.test_dataset = data_file[args.dataset_name]['test']
    # args.test_dataset = ARTS_file[args.dataset_name]['adv_3']

    # datasets = ['lap14', 'res14', 'res15', 'res16']
    # ARTS_datasets = ['lap14', 'res14']
    # for data in ARTS_datasets:
    #     args.dataset_name = data
    #     args.train_dataset = data_file[args.dataset_name]['train']
    #     args.test_dataset = ARTS_file[args.dataset_name]
    #     ins = Instructor(args)
    #     ins.run()

    ins = Instructor(args)
    ins.run()


if __name__ == '__main__':
    main()
