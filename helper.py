import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    print(mrrs['lhs'])
    print(mrrs['rhs'])
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}

def draw_loss(loss_list, save_path):
    if len(loss_list) is not 0:
        x = np.arange(0, len(loss_list), 1)
        plt.plot(x, loss_list)
        plt.savefig(save_path + 'train_loss.jpg')
        plt.cla()

