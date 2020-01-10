
# coding: utf-8
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter

def save_loss(loss, save_dir):
    loss_path = os.path.join(save_dir, 'loss.png')
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax.set_title('Loss')
    ax.plot(range(len(loss)), loss, c='r')
    ax.set_ylabel('loss value', fontsize= 15)
    ax.set_xlabel('epoch', fontsize= 15)
    plt.savefig(loss_path)
    plt.close()

def save_distribution(z, ground_truth, save_dir, epoch):
    ab_idx = np.where(ground_truth == 1)
    n_idx = np.where(ground_truth == 0)
    ab_set = z[ab_idx]
    n_set = z[n_idx]
    show_distribution_3D(save_dir, epoch, n_set[:,0], n_set[:,1], n_set[:,-1], ab_set[:,0], ab_set[:,1], ab_set[:,-1])

def show_distribution_3D(save_dir, epoch, x1, y1, z1, x2=[], y2=[], z2=[]):
    abnormal_path = os.path.join(save_dir, 'epoch{}_abnormal.png'.format(epoch))
    distribution_path = os.path.join(save_dir, 'epoch{}_distribution.png'.format(epoch))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if x2!=[] and y2!=[] and z2!=[]:
        ax.scatter(x2, y2, z2, c='r')
        plt.savefig(abnormal_path)
    ax.scatter(x1, y1, z1, c='b')

    plt.savefig(distribution_path)
    plt.close(fig)