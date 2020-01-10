# coding: utf-8
import os
import copy

import pwlf
import numpy as np
import tensorflow as tf 
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler

def data_loader(dataroot, dataset, splitnum):
    if dataset == 'tlecg':
        normal_label = 1
        train_samples = np.loadtxt(os.path.join(dataroot, dataset, 'tlecg_train.tsv'))
        test_samples = np.loadtxt(os.path.join(dataroot, dataset, 'tlecg_test.tsv'))
    elif dataset == 'faceucr':
        normal_label = 13
        train_samples = np.loadtxt(os.path.join(dataroot, dataset, 'faceucr_train.tsv'))
        test_samples = np.loadtxt(os.path.join(dataroot, dataset, 'faceucr_test.tsv'))
    elif dataset == 'pptw':
        normal_label = 8
        train_samples = np.loadtxt(os.path.join(dataroot, dataset, 'pptw_train.tsv'))
        test_samples = np.loadtxt(os.path.join(dataroot, dataset, 'pptw_test.tsv'))
    elif dataset == 'medicalimage':
        normal_label = 10
        train_samples = np.loadtxt(os.path.join(dataroot, dataset, 'medical_train.tsv'))
        test_samples = np.loadtxt(os.path.join(dataroot, dataset, 'medical_test.tsv'))
    else:
        raise Exception("Dataset {} is not accepted. Only 'tlecg', 'faceucr', 'pptw' and 'medicalimage' is accepted.".format(dataset))
    
    train_labels = train_samples[:, 0]
    train_data = train_samples[:, 1:]
    test_labels = test_samples[:, 0]
    test_data = test_samples[:, 1:]

    train_data = train_data[train_labels == normal_label]
    train_labels = train_labels[train_labels == normal_label]

    # transform normal labels to -2
    train_labels[train_labels == normal_label] = -2
    test_labels[test_labels == normal_label] = -2

    # split data to segments
    train_segdata, train_segnum_list = split_data(train_data, splitnum)
    test_segdata, test_segnum_list = split_data(test_data, splitnum)
    
    loader = {'train_segdata': train_segdata, 'train_labels':train_labels, 'train_segnum_list':train_segnum_list,
              'test_segdata': test_segdata, 'test_labels':test_labels, 'test_segnum_list': test_segnum_list}
    return loader

def split_data(raw_data, split):
    data = []
    if split == 1:
        return raw_data, [split]*len(raw_data)
    # process bar for splitting data
    with tqdm(total= len(raw_data)) as pbar:
        for i, line in enumerate(raw_data):
            my_pwlf = pwlf.PiecewiseLinFit(range(len(line)), line)
            breaks = my_pwlf.fit(split)
            
            for i in range(len(breaks)-1):
                data.append(line[int(breaks[i]):int(breaks[i+1])+1])
            
            pbar.update(1)
            pbar.set_description('Splitting data')
            
    return data, [split] * len(raw_data)

def calculate_score(labels, energy, segnum_list):
    ground_truth = copy.deepcopy(labels)
    ground_truth[ground_truth != -2] = 1
    ground_truth[ground_truth == -2] = 0
    
    y_pred = []
    j = 0
    for seg_num in segnum_list:
        pred = np.max(energy[j: j+ seg_num])
        j = j + seg_num
        y_pred.append(pred)
    energy = y_pred
                        
    return energy, ground_truth

def check_energy(energy_list, label_list, epoch, save_dir):
    energy_list = np.array(energy_list)
    label_list = np.array(label_list)

    labels = np.unique(label_list)

    energy_file = os.path.join(save_dir, 'train_energy.txt')
    with open(energy_file, 'a+') as ef:
        ef.write('============= energy of clusters =============\n')
        for label in labels:
            tmp_energy = energy_list[label_list == label]
            average = np.mean(tmp_energy)
            std = np.std(tmp_energy)
            ef.write('energy of cluster {}: mean/std: {:.5f}/{:.5f}\t--epoch {}\n'.format(label, average, std, epoch))
        ef.write('==============================================\n\n')

def evaluate(ground_truth, predict):
    if type(ground_truth) != np.ndarray:
        ground_truth = np.array(ground_truth)
    if type(predict) != np.ndarray:
        predict = np.array(predict)
    _, _, thresholds = roc_curve(ground_truth, predict, pos_label= 1)
    roc_auc = roc_auc_score(ground_truth, predict)
    
    best_f1, best_pre, best_recall, best_th = 0, 0, 0, 0
    for threshold in thresholds:
        tmp_label = predict.copy()
        tmp_label[tmp_label >= threshold] = 1
        tmp_label[tmp_label < threshold] = 0
        tmp_f1 = f1_score(ground_truth, tmp_label, pos_label= 1)
        if tmp_f1 > best_f1:
            best_f1 = tmp_f1
            best_th = threshold
            best_pre = precision_score(ground_truth, tmp_label, pos_label= 1)
            best_recall = recall_score(ground_truth, tmp_label, pos_label= 1)
    return roc_auc, best_f1, best_th, best_pre, best_recall

def make_batch(inputs, max_sequence_length = None, normalize = False):     
    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        inputs = scaler.fit_transform(inputs)           
                    
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    
    # zero matrix of shape [batch_size, max_sequence_length]
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.float64)
    
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major#.swapaxes(0, 1)
    
    return inputs_time_major, sequence_lengths

def reconstruction_distances(input_tensor, reconstruction):
    with tf.variable_scope('reconstruction_distances'):
        squared_x = tf.reduce_sum(tf.square(input_tensor),
                                  name='squared_x',
                                  axis=1) + 1e-12
        # Relative distance
        input_tensor = input_tensor[:,:,0]
        reconstruction = reconstruction[:,:,0]
        dist = tf.norm(input_tensor - reconstruction, ord=2, axis=1, keepdims=True, name='dist')
        relative_dist = dist / tf.norm(input_tensor, ord=2, axis=1, keepdims=True, name='relative_dist')                                  
                                          
        # Cosine similarity
        n1 = tf.nn.l2_normalize(input_tensor,1)
        n2 = tf.nn.l2_normalize(reconstruction,1)
        cosine_similarity = tf.reduce_sum(tf.multiply(n1, n2), 1, keepdims=True, name='cosine_similarity')
        return squared_x, relative_dist, cosine_similarity, dist