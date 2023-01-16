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
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}

def draw_loss(loss_list, save_path):
    if len(loss_list) is not 0:
        x = np.arange(0, len(loss_list), 1)
        plt.plot(x, loss_list)
        plt.savefig(save_path + 'train_loss.jpg')
        plt.cla()

def draw_score(score, target, hq_o_id, lq_o_id, mid_o_id, dataset, x_label, y_label, beta, mode='Score'):
    """
        Function to plot

        Parameters
        ----------
        score: Tensor[n_entitiies]
        target: 目标实体的id
        hq_o_id: 高质量实体id
        lq_o_id: 低质量实体id
        mid_o_id: 中质量实体id
        dataset: dataset类
        x_label: x轴标签
        y_label: y轴标签
        mode: 'Score'表示画分数图，否则画权重图

        Returns
        -------
        """
    if mode.lower() =='score':
        print_weights(score, hq_o_id, lq_o_id, mid_o_id, dataset, x_label, y_label)
    elif mode=='weight':
        score_cl = score.clone()
        p_no_mask = torch.softmax(0.2 * (score_cl), dim=0)
        y = p_no_mask
        print_weights(y, hq_o_id, lq_o_id, mid_o_id, dataset, x_label, y_label)
    else:
        thre_value = score.max() * 0.2
        score_cl =score.clone()
        score_cl[score_cl < thre_value] = beta
        p_mask =  torch.softmax(0.2 * (score_cl), dim=0)
        y = p_mask
        y[target] = p_mask.min()
        print_weights(y, hq_o_id, lq_o_id, mid_o_id, dataset, x_label, y_label)

def print_query(q, dataset):
    if q[1] < len(dataset.id2name['id2rel']):
        s = dataset.id2name['id2ent'][q[0].item()]
        p = dataset.id2name['id2rel'][q[1].item()]
        o = dataset.id2name['id2ent'][q[2].item()]
        t = dataset.id2name['id2time'][q[3].item()]
        print('predict objects')
    else:
        s = dataset.id2name['id2ent'][q[2].item()]
        p = dataset.id2name['id2rel'][(q[1].item() - len(dataset.id2name['id2rel']))]
        o = dataset.id2name['id2ent'][q[0].item()]
        t = dataset.id2name['id2time'][q[3].item()]
        print('predict subjects')
    return print('({0},{1},{2},{3})'.format(s,p,o,t))

def print_candidates(candidates, dataset):
    ent_name = [dataset.id2name['id2ent'][i.item()] for i in candidates]
    return print(ent_name)

def print_weights(score, hq_o_id, lq_o_id, mid_o_id, dataset, x_label, y_label):
    # id转string
    hq_o_name = dataset.id2name['id2ent'][hq_o_id]
    lq_o_name = dataset.id2name['id2ent'][lq_o_id]
    mid_o_name = dataset.id2name['id2ent'][mid_o_id]
    # 画线
    x = np.arange(0, score.shape[0], 1)
    y = score.detach().cpu().numpy()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, y, color='cornflowerblue')
    # 画点
    plt.plot([hq_o_id], [score[hq_o_id]], 'o', color = 'r', label = hq_o_name)
    plt.text(hq_o_id*1.05, score[hq_o_id], hq_o_name, color =  'r', fontsize=12, style = "italic", weight = "medium" )
    plt.plot([mid_o_id], [score[mid_o_id]], 'o', color='deeppink', label=mid_o_name)
    plt.text(mid_o_id * 1.1, score[mid_o_id], mid_o_name, color='deeppink', fontsize=12, style="italic", weight="medium")
    plt.plot([lq_o_id], [score[lq_o_id]], 'o', color = 'forestgreen', label = lq_o_name)
    plt.text(lq_o_id*1.1, score[lq_o_id], lq_o_name, color =  'forestgreen', fontsize=12, style = "italic", weight = "medium" )
    plt.show()



