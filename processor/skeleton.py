#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight.torchlight as torchlight
from torchlight.torchlight  import str2bool
from torchlight.torchlight  import DictAction
from torchlight.torchlight  import import_class

# import torchlight as torchlight 
# from torchlight  import str2bool
# from torchlight  import DictAction
# from torchlight  import import_class

from .densenet_processor import Processor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class SK_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        # self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.epoch_info['top{}'.format(k)]=100*accuracy
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))
    def show_confusion_matrix(self):
        instance_num, class_num = self.result.shape
        matrix = np.zeros((class_num,class_num))
        rank = self.result.argsort()
        for i in range(instance_num):
            t = self.label[i] # true label
            p = rank[i,-1]    # predict label
            matrix[t,p]+=1
        matrix_log = ''
        for row in matrix:
            matrix_log += ' '.join(['%3d'%(num) for num in row]) + '\n'
        matrix_log = 'Confusion Matrix is:\n' + matrix_log
        self.io.print_log(matrix_log)

    def show_topk_by_category(self,top_k):
        instance_num, class_num = self.result.shape
        rank = self.result.argsort()
        hit_top_k = [[] for i in range(class_num)]
        for i in range(instance_num):
            l = self.label[i]
            hit_top_k[l].append(l in rank[i, -top_k:])

        accuracy_list = []
        for hit_per_category in hit_top_k:
            if hit_per_category:
                accuracy_list.append(sum(hit_per_category) * 1.0 / len(hit_per_category))
            else:
                accuracy_list.append(0.0)
        log='\tTOP{} By Category:'.format(top_k) + '\t'.join(['{:.2f}'.format(x * 100) for x in accuracy_list])
        self.io.print_log(log)
    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, label in loader:

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output = self.model(data)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:
            
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(data)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss']= np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k)
                self.show_topk_by_category(k)
            self.show_confusion_matrix()

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Densenet Skeleton Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
