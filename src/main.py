# coding: utf-8

import os

from numpy.random import seed
import tensorflow as tf
from tensorflow import set_random_seed
from args import Parser
from utils import data_loader
from Seq2Cluster_test import Seq2Cluster

seed(1)
set_random_seed(2) 

opts = Parser().parse()

os.environ["CUDA_VISIBLE_DEVICES"] = opts['gpu']
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

dataloader = data_loader(opts['datapath'], opts['dataset'], opts['split'])

tf.reset_default_graph()

model = Seq2Cluster(opts)

if opts['test'] is False:
    model.train(dataloader['train_segdata'],
                dataloader['test_labels'], 
                dataloader['test_segdata'], 
                dataloader['train_segnum_list'],
                dataloader['test_segnum_list'])
else:
    model.test(dataloader['test_segdata'], 
               dataloader['test_labels'], 
               dataloader['test_segnum_list'])