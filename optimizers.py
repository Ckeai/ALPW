# Copyright (c) Facebook, Inc. and its affiliates.

import tqdm
import torch
from torch import nn
from torch import optim

from models import TKBCModel
from regularizers import Regularizer
from datasets import TemporalDataset
bar_format = '{desc}{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining}{postfix}]'
from torch.nn import functional as F
import numpy as np
from helper import draw_score, print_query, print_candidates,print_weights

class TKBCOptimizer(object):
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.lamda = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.lamda.data.fill_(1.0)

    def epoch(self, examples: torch.LongTensor):
        print("WLNS w/o WM and AWNSR")
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        L_FIT = []
        with tqdm.tqdm(total=examples.shape[0], bar_format=bar_format, disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()
                predictions, factors, time = self.model.forward(input_batch)
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                l_reg = self.emb_regularizer.forward(factors)
                l_time = torch.zeros_like(l_reg)
                if time is not None:
                    l_time = self.temporal_regularizer.forward(time)
                l = l_fit + l_reg + l_time

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(
                    loss=f'{l_fit.item():.3f}',
                    reg=f'{l_reg.item():.3f}',
                    cont=f'{l_time.item():.3f}'
                )
                L_FIT.append(l_fit.item())
            return np.mean(L_FIT)

    def epoch_pretrain(self, examples: torch.LongTensor):
        print("WLNS w/o WM")
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        L_FIT = []
        with tqdm.tqdm(total=examples.shape[0], bar_format=bar_format, disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                              b_begin:b_begin + self.batch_size
                              ].cuda()
                batch = input_batch.shape[0]

                predictions, factors, time = self.model.forward(input_batch)
                truth = input_batch[:, 2]
                label = torch.zeros_like(predictions)
                label.scatter_(dim=1, index=truth.unsqueeze(0).t().cuda(), src=torch.ones((batch, 1)).cuda())
                neg_score_no_soft = predictions[label == False].reshape(batch, -1)
                '''log-softmax后的正样本分数'''
                pos_score = F.log_softmax(predictions, dim=1)[label == True]
                '''log-softmax后的负样本分数'''
                neg_score = (F.log_softmax(1 - predictions, dim=1)[label == False]).reshape(batch, -1)
                pos_loss = pos_score.sum()
                '''只对原负样本求权重'''
                p = torch.softmax(0.2 * (neg_score_no_soft), dim=1)
                neg_loss = (p * neg_score).sum()
                l_fit = -(pos_loss +  neg_loss) / 2 / batch
                l_reg = self.emb_regularizer.forward(factors)
                l_time = torch.zeros_like(l_reg)
                if time is not None:
                    l_time = self.temporal_regularizer.forward(time)
                l = l_fit + l_reg + l_time

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(
                    loss=f'{l_fit.item():.3f}',
                    reg=f'{l_reg.item():.3f}',
                    cont=f'{l_time.item():.3f}'
                )
                L_FIT.append(l_fit.item())
            return np.mean(L_FIT)

    '''Adaptively learn the threshold to weaken low-quality negative samples'''
    def WM_epoch(self, dataset, examples: torch.LongTensor, beta, temp, alpha):
        print("use threshold to weaken low quality sample")
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        L_FIT = []
        with tqdm.tqdm(total=examples.shape[0], bar_format=bar_format, disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                              b_begin:b_begin + self.batch_size
                              ].cuda()
                batch = input_batch.shape[0]

                predictions, factors, time = self.model.forward(input_batch)
                truth = input_batch[:, 2]
                label = torch.zeros_like(predictions)
                label.scatter_(dim=1, index=truth.unsqueeze(0).t().cuda(), src=torch.ones((batch, 1)).cuda())
                neg_score_no_soft = predictions[label == False].reshape(batch, -1)
                '''log-softmax后的正样本分数'''
                pos_score = F.log_softmax(predictions, dim=1)[label == True]
                '''log-softmax后的负样本分数'''
                neg_score = (F.log_softmax(1 - predictions, dim=1)[label == False]).reshape(batch, -1)
                pos_loss = pos_score.sum()
                '''只对原负样本求权重'''
                thre_value = neg_score_no_soft.max(1).values*alpha
                neg_score_no_soft[neg_score_no_soft< thre_value.unsqueeze(1)] = beta
                p = torch.softmax( temp*(neg_score_no_soft), dim=1)
                neg_loss = (p * neg_score).sum()
                l_fit = -(pos_loss + neg_loss) / 2 / batch
                l_reg = self.emb_regularizer.forward(factors)
                l_time = torch.zeros_like(l_reg)
                if time is not None:
                    l_time = self.temporal_regularizer.forward(time)
                l = l_fit + l_reg + l_time

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                # self.scheduler_1.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(
                    loss=f'{l_fit.item():.3f}',
                    reg=f'{l_reg.item():.3f}',
                    cont=f'{l_time.item():.3f}'
                )
                L_FIT.append(l_fit.item())
            return np.mean(L_FIT)
