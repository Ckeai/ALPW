import argparse
import logging
import torch
from torch import optim

from datasets import TemporalDataset
from optimizers import TKBCOptimizer
from models import TComplEx
from regularizers import N3, Lambda3
import os
import sys
import time
import numpy as np
import random
from helper import avg_both, draw_loss


class Runner():
    def __init__(self, param):
        self.p = param
        self.dataset = TemporalDataset(self.p.dataset)
        self.modelname = self.p.model
        self.sizes = self.dataset.get_shape()
        self.emb_reg = N3(args.emb_reg)
        self.time_reg = Lambda3(args.time_reg)
        self.model = self.add_model()
        self.opt = self.add_optimizer(self.model.parameters())
        self.beta = self.p.beta
        self.temp = self.p.temp
        self.alpha = self.p.alpha
        '''file storage path'''
        self.savename = self.p.model + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime(
            '%H:%M:%S')
        self.root = 'results/'  + self.p.dataset + '/' + self.modelname
        self.PATH = os.path.join(self.root,
                                 'rank{:.0f}/lr{:.4f}/batch{:.0f}/emb_reg{:.5f}/time_reg{:.5f}/'.format(self.p.rank,
                                                                                                        self.p.learning_rate,
                                                                                                        self.p.batch_size,
                                                                                                        self.p.emb_reg,
                                                                                                        self.p.time_reg))
        try:
            os.makedirs(self.PATH)
        except FileExistsError:
            pass

    def add_model(self):
        """
        Creates the computational graph

        Parameters
            ----------
            model_name:     Contains the model name to be created

            Returns
            -------
            Creates the computational graph for model and initializes it
        """
        if self.modelname.lower() == 'tcomplex':
            model = TComplEx(self.sizes, self.p.rank, no_time_emb=self.p.no_time_emb)
        else:
            raise NotImplementedError
        model = model.cuda()
        return model

    def add_optimizer(self, parameters):
        return optim.Adagrad(parameters, lr=self.p.learning_rate)

    def save_model(self, save_path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

        Parameters
        ----------
        save_path: path where the model is saved

        Returns
        -------
        """
        state = {
            'state_dict': self.model.state_dict(),
            'best_val': self.best_val,
            'best_epoch': self.best_epoch,
            'optimizer': self.opt.state_dict(),
            'args': vars(self.p)
        }
        torch.save(state, save_path)

    def load_model(self, load_path):
        """
        Function to load a saved model

        Parameters
        ----------
        load_path: path to the saved model

        Returns
        -------
        """
        state = torch.load(load_path)
        state_dict = state['state_dict']
        self.best_val = state['best_val']
        self.best_val_mrr = self.best_val['MRR']
        print("loading model........")
        print(state['args'])
        self.model.load_state_dict(state_dict)
        self.opt.load_state_dict(state['optimizer'])
        return state['args']

    def fit(self):
        self.patience = 0
        self.best_val = {'MRR': 0., 'hits@[1,3,10]': 0.}
        self.best_epoch = 0
        self.curve = {'train': []}
        for epoch in range(self.p.max_epochs):
            print("[ Epoch:", epoch, "]")
            examples = torch.from_numpy(self.dataset.get_train().astype('int64'))
            self.model.train()
            optimizer = TKBCOptimizer(self.model, self.emb_reg, self.time_reg, self.opt, batch_size=self.p.batch_size)
            epoch_loss = optimizer.epoch(examples, self.beta, self.temp, self.alpha)
            self.curve['train'].append(epoch_loss)
            if epoch < 0 or (epoch + 1) % self.p.valid_freq == 0:
                if self.dataset.interval:
                    valid, test = [avg_both(*self.dataset.eval(self.model, split, -1)) for split in ['valid', 'test']]
                    print("valid:", valid['MRR'])
                    print("test:", test['MRR'])
                else:
                    valid, test, train = [
                        avg_both(*self.dataset.eval(self.model, split, -1 if split != 'train' else 50000)) for split in
                        ['valid', 'test', 'train']]
                    print("valid: ", valid['MRR'])
                    print("test: ", test['MRR'])
                    print("train: ", train['MRR'])
                f = open(os.path.join(self.PATH, 'result_{}.txt'.format(self.savename)), 'w+')
                f.write("\n VALID: ")
                f.write(str(valid))
                f.write("\n\nconfig : ")
                f.write(list_of_arguments)
                f.write("\n Best_epoch:")
                f.write(str(self.best_epoch))
                f.close()
                # early-stop with patience
                if valid['MRR'] < self.best_val['MRR']:
                    self.patience += 1
                    if self.patience >= self.p.drop_time:
                        print("Early stopping ...")
                        break
                else:
                    self.patience = 0
                    self.best_val = valid
                    self.best_epoch = epoch
                    self.save_model(os.path.join(self.PATH, self.savename))
                if not self.dataset.interval:
                    print("\t TRAIN: ", train)
                print("\t VALID : ", valid)
                print("best epoch:{}".format(self.best_epoch))
        self.argu = self.load_model(os.path.join(self.PATH, self.savename))
        results = avg_both(*self.dataset.eval(self.model, 'test', -1))
        total_time = (time.time() - start_time) / 60.0
        print("Total execution time:{:4f}".format(total_time))
        print("\n\nTEST : ", results)
        print("best epoch:{}".format(self.best_epoch))
        f = open(os.path.join(self.PATH, 'result_{}.txt'.format(self.savename)), 'w+')
        f.write("\n TEST: ")
        f.write(str(results))
        f.write("\n\nconfig : ")
        f.write(str(self.argu))
        f.write("\n Best_epoch:")
        f.write(str(self.best_epoch))
        f.write("\n Total execution time:")
        f.write(str(total_time))
        f.close()
        draw_loss(self.curve['train'], self.PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Temporal ComplEx"
    )
    parser.add_argument(
        '--dataset', type=str,
        help="Dataset name"
    )
    models = [
        'ComplEx', 'TComplEx', 'TNTComplEx'
    ]
    parser.add_argument(
        '--model', choices=models,
        help="Model in {}".format(models)
    )

    parser.add_argument(
        '--savename', default=TComplEx, type=str,
        help="Model in {}".format(models)
    )
    parser.add_argument(
        '--max_epochs', default=1000, type=int,
        help="Number of epochs."
    )
    parser.add_argument(
        '--valid_freq', default=5, type=int,
        help="Number of epochs between each valid."
    )
    parser.add_argument(
        '--rank', default=100, type=int,
        help="Factorization rank."
    )
    parser.add_argument(
        '--beta', default=-1, type=int,
        help="assign beta to low-quality samples."
    )
    parser.add_argument(
        '--temp', default=0.2, type=float,
        help="tempture factor."
    )
    parser.add_argument(
        '--batch_size', default=1000, type=int,
        help="Batch size."
    )
    parser.add_argument(
        '--alpha', default=0.2, type=float,
        help="tau * high_score = threshold"
    )
    parser.add_argument(
        '--learning_rate', default=1e-1, type=float,
        help="Learning rate"
    )
    parser.add_argument(
        '--emb_reg', default=0., type=float,
        help="Embedding regularizer strength"
    )
    parser.add_argument(
        '--time_reg', default=0., type=float,
        help="Timestamp regularizer strength"
    )
    parser.add_argument(
        '--no_time_emb', default=False, action="store_true",
        help="Use a specific embedding for non temporal relations"
    )
    parser.add_argument(
        '--drop_time', default=10, type=int,
        help="the number of epochs on the validation set does not exceed the previous optimal value"
    )
    args = parser.parse_args()

    list_of_arguments = '\t'.join(sys.argv)

    # fixed seed
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    random.seed(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    learner = Runner(args)
    start_time = time.time()
    learner.fit()
