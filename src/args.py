# coding: utf-8

import argparse

class Parser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # Base
        self.parser.add_argument('--pad', type=int, default=0, help='padding of data')
        self.parser.add_argument('--eos', type=int, default=1, help='<EOS> of sequence')
        self.parser.add_argument('--vocab_size', type=int, default=1)

        # Data
        self.parser.add_argument('--datapath', type=str, default= './dataset', help='directory of dataset')
        self.parser.add_argument('--dataset', type=str, default='tlecg', help='dataset name')
        self.parser.add_argument('--nmix', type=int, help='number of component mixture in GMM')
        self.parser.add_argument('--split', type=int, help='number of data segements')

        # Train and Test
        self.parser.add_argument('--test', action='store_true', help='test model if use this param')
        self.parser.add_argument('--endim', type=int, default=15, help='dimension of hidden vector of encoder(encoder hidden units)')
        self.parser.add_argument('--lambda', type=float, default=1e-4, help='hyper parameter lambda')
        self.parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        self.parser.add_argument('--epoch', type=int, default= 2001, help='number of epochs when training')
        self.parser.add_argument('--gpu', type=str, default='1', help='gpu ids')
        self.parser.add_argument('--nkeep', type=int, default=25, help='maximum number of checkpoints to keep')
        self.parser.add_argument('--outdir', type=str, default='./output', help='directory to store result')
        self.parser.add_argument('--nsave', type=int, default=100, help='save checkpoint every `nsave` epochs')

        self.opt = None
    
    def parse(self):
        self.opt = self.parser.parse_args()

        self.opt = vars(self.opt)

        self.opt['encoder_hidden_units'] = self.opt['endim']
        self.opt['decoder_hidden_units'] = self.opt['endim']
        self.opt['GMM_input_dim'] = self.opt['endim'] + 2
        self.opt['num_dynamic_dim'] = self.opt['endim']
        print('============arguments===========')
        print(self.opt)
        print('================================')

        return self.opt     
