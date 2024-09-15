import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from sklearn import metrics
import torch.nn.functional as F

from bucket_iterator import BucketIterator
from data_utils import Bert_ABSA_Dataset, ABSA_collate_fn, ABSA_Dataset_Reader
from model.CTS_BERT import CTS_BERT


class Instructor:
    def __init__(self, args):
        self.args = args
        self.train_dataloader = None
        if 'BERT' in args.model_name:
            time.sleep(5)
            bert = BertModel.from_pretrained('bert-base-uncased')
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            print('preparing {0} dataset ...'.format(args.dataset_name))
            train_data = Bert_ABSA_Dataset(args, args.train_dataset, bert_tokenizer)
            test_data = Bert_ABSA_Dataset(args, args.test_dataset, bert_tokenizer)
            self.train_dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, collate_fn=ABSA_collate_fn)
            self.test_dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size, collate_fn=ABSA_collate_fn)
            self.model = args.model(args, bert).to(args.device)
        else:   # GloVe
            absa_dataset = ABSA_Dataset_Reader(args)
            self.train_dataloader = BucketIterator(data=absa_dataset.train_data, batch_size=args.batch_size, shuffle=True)
            self.test_dataloader = BucketIterator(data=absa_dataset.test_data, batch_size=args.batch_size)
            self.model = args.model(args, absa_dataset.embedding_matrix).to(args.device)

        self._print_args()
        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=args.device.index))

    def _train(self, optimizer, max_test_acc_overall=0):
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        model_path = ''
        for epoch in range(self.args.num_epoch):
            print('*' * 100)
            print('epoch: {}'.format(epoch))
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                global_step += 1
                self.model.train()
                optimizer.zero_grad()    # 梯度清0
                inputs = (i.to(self.args.device) for i in sample_batched)
                outputs = self.model(inputs)
                targets = sample_batched[-1].to(self.args.device)
                loss = F.cross_entropy(outputs, targets, reduction='mean')   # 计算输出与实际的偏差
                loss.backward()   # 梯度下降
                optimizer.step()    # 更新优化器
                if global_step % self.args.log_step == 0:    # 每过几步就再测试集上训练一下
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total
                    test_acc, f1 = self._evaluate()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            model_path = './save_model/{}_{}_acc_{:.4f}_f1_{:.4f}'.format(self.args.model_name, self.args.dataset_name, test_acc, f1)
                            self.best_model = copy.deepcopy(self.model)
                            print('>>> save model: {}'.format(model_path))
                    if f1 > max_f1:
                        max_f1 = f1
                    print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc, test_acc, f1))
        return max_test_acc, max_f1, model_path

    def _test(self):
        self.model = self.best_model
        # bert = BertModel.from_pretrained('bert-base-uncased')
        # self.model = CTS_BERT(args=self.args, bert=bert).to(self.args.device)
        # absa_dataset = ABSA_Dataset_Reader(self.args)
        # self.test_dataloader = BucketIterator(data=absa_dataset.test_data, batch_size=self.args.batch_size)
        # self.model = self.args.model(self.args, absa_dataset.embedding_matrix).to(self.args.device)
        # temp = torch.load('./save_model/CTS_BERT_res14_acc_0.8696_f1_0.8103')
        # self.model.load_state_dict(state_dict=temp)
        self.model.eval()
        test_report, test_confusion, acc, f1 = self._evaluate(show_results=True)
        print('acc = {0}, f1 = {1}'.format(acc, f1))
        print('=========================')
        print("Precision, Recall and F1-Score...")
        print(test_report)
        print("Confusion Matrix...")
        print(test_confusion)

    def _evaluate(self, show_results=False):
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        targets_all, outputs_all = None, None
        with torch.no_grad():
            for batch, sample_batched in enumerate(self.test_dataloader):
                inputs = (i.to(self.args.device) for i in sample_batched)
                targets = sample_batched[-1].to(self.args.device)
                outputs = self.model(inputs)
                n_test_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_test_total += len(outputs)
                targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
                outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')

        labels = targets_all.data.cpu()
        predict = torch.argmax(outputs_all, -1).cpu()
        if show_results:
            report = metrics.classification_report(labels, predict, digits=4)
            confusion = metrics.confusion_matrix(labels, predict)
            return report, confusion, test_acc, f1

        return test_acc, f1

    def get_bert_optimizer(self, model):
        if 'BERT' in self.args.model_name:
            bert_model = model.bert
            bert_params_dict = list(map(id, bert_model.parameters()))

            if self.args.Inter_aspect:
                bert_model_2 = model.AA_encoder.bert_model
                bert_params_dict_2 = list(map(id, bert_model_2.parameters()))
                bert_params_dict += bert_params_dict_2

            base_params = filter(lambda p: id(p) not in bert_params_dict, model.parameters())
            optimizer_grouped_parameters = [
                {"params": [p for p in base_params if p.requires_grad]},
                {"params": [p for p in bert_model.parameters() if p.requires_grad], "lr": self.args.bert_lr}
            ]

            optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.args.bert_lr, weight_decay=self.args.l2reg)
        else:
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = torch.optim.Adam(_params, lr=self.args.glove_learning_rate, weight_decay=self.args.l2reg)

        return optimizer

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.args):
            print('>>> {0}: {1}'.format(arg, getattr(self.args, arg)))

    def run(self):
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.get_bert_optimizer(self.model)   # 获得bert的优化器
        max_test_acc_overall = 0    # 全局的最大正确率
        max_f1_overall = 0    # 最大f1值
        max_test_acc, max_f1, model_path = self._train(optimizer, max_test_acc_overall)  # 把损失函数和优化器一起传入训练，返回最大正确率和f1值
        print('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
        max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
        max_f1_overall = max(max_f1, max_f1_overall)
        torch.save(self.best_model.state_dict(), model_path)   # torch保存最终结果
        print('>> saved: {}'.format(model_path))
        print('#' * 60)
        print('max_test_acc_overall:{}'.format(max_test_acc_overall))
        print('max_f1_overall:{}'.format(max_f1_overall))
        self._test()
