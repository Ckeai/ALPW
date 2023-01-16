import parser

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib import colors

def plot_curve(path, name, x, y, curve_color, marker_, line_, x_label, y_label):
    plt.plot(x, y, color=curve_color, linestyle=line_, marker=marker_, linewidth=2.0)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(os.path.join(path, name), dpi=600)
    plt.show()

if __name__ == "__main__":
    path_dir = "./new_results/pretained/"
    x={'rank':np.array([20, 50, 100, 200, 500, 1000, 2000]),
       'beta':np.array([-50, -20, -10, -5, -2, -1, 0]),
       'alpha': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
       'temp': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1])
       }
    ICEWS14_y={'rank':np.array([0.4456, 0.5021, 0.5408, 0.5720, 0.6078, 0.6228, 0.6312]),
                  'beta':np.array([0.6064, 0.6284, 0.6300, 0.6306, 0.6299, 0.6301, 0.6290]),
                  'alpha':np.array([0.6302, 0.6309, 0.6307, 0.6302, 0.6302,0.6301,0.6294,0.6294]),
                  'temp':np.array([0.6289, 0.6308, 0.6304, 0.6295, 0.6290, 0.6300,0.6293,0.6282, 0.6284, 0.6281])
                  }
    ICEWS05_y = {'rank': np.array([0.4767, 0.5302, 0.5715, 0.6118, 0.6548, 0.6734, 0.6820]),
                      'beta': np.array([0.6778, 0.6797, 0.6815, 0.6817, 0.6819, 0.6818, 0.6817]),
                      'alpha':np.array([0.6816, 0.6820, 0.6819, 0.6818, 0.6817, 0.6817, 0.6815, 0.6814]),
                      'temp':np.array([0.6815, 0.6819, 0.6816, 0.6810, 0.6805, 0.6797, 0.6798,0.6793, 0.6795, 0.6790])
                    }
    plot_format = {'rank': 'o', 'beta':'D', 'alpha':'s', 'temp':'2'}
    plot_color = {'rank': 'cornflowerblue', 'beta':'lightcoral', 'alpha':'mediumpurple', 'temp':'orange'}
    plot_line = {'rank': '-', 'beta':'-', 'alpha':'-', 'temp':'-'}
    x_label = {'rank': 'Dimension k', 'beta': r'$\beta$', 'alpha': r'$\alpha$', 'temp': r'$\tau$'}
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ICEWS14', 'ICEWS05-15'], type=str, help="dataset")
    parser.add_argument('--param', choices=['rank','beta','alpha','temp'], type=str, help="dataset")
    args = parser.parse_args()

    s_x = x[args.param]
    if args.dataset.lower() == 'icews14':
        s_y = ICEWS14_y[args.param]
    else:
        s_y = ICEWS05_y[args.param]
    if args.param =="alpha":
        if args.dataset.lower()=="icews14":
            plt.ylim(0.6290, 0.6315)
        else:
            plt.ylim(0.6810, 0.6823)
    if args.param == "rank":
        if args.dataset.lower() == "icews05-15":
            plt.ylim(0.46, 0.695)
    print("Preparing dataset {}".format(args.dataset))
    print("Preparing parameter {}".format(args.param))
    path = os.path.join(path_dir, args.dataset)
    plot_curve(path, args.param+'_'+args.dataset, s_x, s_y, plot_color[args.param], plot_format[args.param], plot_line[args.param],x_label[args.param] , 'MRR')