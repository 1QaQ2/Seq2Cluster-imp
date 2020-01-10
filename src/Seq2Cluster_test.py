# coding: utf-8

import re
import os
import math

import tensorflow as tf
import numpy as np
from time import time

from plot import save_distribution, save_loss
from utils import make_batch, check_energy
from utils import evaluate, calculate_score


class Seq2Cluster(object):
    
    def __init__(self, opts):
        self.opts = opts
        self.is_training = tf.placeholder_with_default(tf.constant(True), [], name='is_training')
        
        self._create_network()
        self._create_loss_optimizer()
        self.savers()
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

              
    def _create_network(self):
        # to be updated
        pass
        
    def _create_loss_optimizer(self):
        # to be updated
        pass
    
    def savers(self):
        # to be updated

        self.model_dir = os.path.join(self.opts['outdir'], self.opts['dataset'], 'checkpoint')
        self.plot_dir = os.path.join(self.opts['outdir'], self.opts['dataset'], 'img', 'Mix{}_Encoder{}_Split{}'.format(self.opts['nmix'], self.opts['encoder_hidden_units'], self.opts['split']))
        self.save_dir = os.path.join(self.opts['outdir'], self.opts['dataset'])
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    
    def reconstruction_distances(self, input_tensor, reconstruction):
        # to be updated
        pass
    
    def compute_likelihood(self, sample_z, pi_input, mu, sigma):
        x_t = tf.reshape(sample_z, shape=[-1, 1, self.opts['GMM_input_dim']])
        x_t = tf.tile(x_t, [1, self.opts['nmix'], 1])

        mixture_mean = tf.cast(mu, dtype=tf.float64)

        det_diag_cov = tf.reduce_prod(sigma, axis=1)
        x_t64 = tf.cast(x_t, dtype=tf.float64)
        x_sub_mean = x_t64 - mixture_mean
        
        z_norm = x_sub_mean ** 2 / sigma
        p = tf.cast(pi_input, dtype=tf.float64)
        t1 = p * tf.reduce_prod(tf.exp(-0.5 * z_norm), axis=2)
        t2 = ((2 * math.pi) ** (0.5 * self.opts['GMM_input_dim'])) * (det_diag_cov ** 0.5)
        tmp = (t1 / t2)

        likelihood = tf.reduce_sum(tmp, 1)
        return likelihood

    def train(self, train_data, test_labels, test_data, train_segnum_list, test_segnum_list, normalize = False):
        # to be updated
        print('The network code and training code will be updated soon...')
        
    def eval(self, test_data, test_segnum_list, normalize = False):
        # to be updated
        pass

    def test(self, test_segdata, test_labels, test_segnum_list, normalize= False):
        EOS = self.opts['eos']
        PAD = self.opts['pad']
        # preparing data for test
        targets = [(sequence.tolist()) + [EOS] + [PAD] * 2 for sequence in test_segdata]
        test_segdata, inputs_length = make_batch(test_segdata, normalize = normalize)
        targets, _ = make_batch(targets, normalize = normalize)
        test_segdata = np.expand_dims(test_segdata, axis=2).swapaxes(0, 1)
        targets = np.expand_dims(targets, axis=2).swapaxes(0, 1)    

        seg_index = []
        for n in test_segnum_list:
            seg_index = seg_index + [i for i in range(n)]

        # restore model from checkpoint
        sess = tf.Session()
        saver = tf.train.import_meta_graph(os.path.join(self.model_dir, 'trained-seq2cluster-best.meta'))
        saver.restore(sess, os.path.join(self.model_dir, 'trained-seq2cluster-best'))
        graph = tf.get_default_graph()

        encoder_inputs = graph.get_tensor_by_name("encoder_inputs:0")
        encoder_inputs_length = graph.get_tensor_by_name("encoder_inputs_length:0")
        frag_index = graph.get_tensor_by_name("fragment_index:0")
        decoder_targets = graph.get_tensor_by_name("decoder_targets:0")
        is_training = graph.get_tensor_by_name("is_training:0")
        gamma = graph.get_tensor_by_name('estimator/predicted_memebership:0')
        z = tf.get_collection('latent_space')
        var_list = tf.get_collection('vars')
        
        feed_d = {
            encoder_inputs: test_segdata,
            encoder_inputs_length: inputs_length,
            frag_index: np.array(seg_index)[:, np.newaxis],
            decoder_targets: targets,
            is_training: False}

        likelihood = self.compute_likelihood(z[0], gamma, var_list[0], var_list[1])

        likelihood, _, _, _= sess.run([likelihood, var_list, z ,gamma], feed_dict=feed_d)
        
        cal_energy = -np.log(likelihood + 1e-12)
        energies, ground_truth = calculate_score(test_labels, cal_energy, test_segnum_list)
        roc_auc, best_f1, best_th, best_pre, best_recall = evaluate(ground_truth, energies)

        print('============ test result for {} ============='.format(self.opts['dataset']))
        print('threshold: {}'.format(best_th))
        print('best f1: {}'.format(best_f1))
        print('precision: {}'.format(best_pre))
        print('recall: {}'.format(best_recall))
        print('roc auc score: {}'.format(roc_auc))
