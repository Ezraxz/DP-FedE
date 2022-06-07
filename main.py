


import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import logging
import os
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import pickle
from dataloader import *
from utils import *
from fede import FedE



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/fb15k237-3.pkl', type=str)
    
    parser.add_argument('--name', default='fb15k237_3', type=str)
    parser.add_argument('--state_dir', '-state_dir', default='./state', type=str)
    parser.add_argument('--log_dir', '-log_dir', default='./log', type=str)
    parser.add_argument('--tb_log_dir', '-tb_log_dir', default='./tb_log', type=str)
    
    parser.add_argument('--num_multi', default=3, type=int)
    parser.add_argument('--model', default='TransE', choices=['TransE', 'RotatE', 'DistMult', 'ComplEx'])
    
    # one task hyperparam
    parser.add_argument('--one_client_idx', default=0, type=int)
    parser.add_argument('--max_epoch', default=10000, type=int)
    parser.add_argument('--log_per_epoch', default=1, type=int)
    parser.add_argument('--check_per_epoch', default=10, type=int)


    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int)
    parser.add_argument('--num_neg', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=int)
    
    # for FedE
    parser.add_argument('--num_client', default=3, type=int)
    parser.add_argument('--max_round', default=10000, type=int)
    parser.add_argument('--local_epoch', default=3)
    parser.add_argument('--fraction', default=1, type=float)
    parser.add_argument('--log_per_round', default=1, type=int)
    parser.add_argument('--check_per_round', default=5, type=int)

    parser.add_argument('--early_stop_patience', default=15, type=int)
    parser.add_argument('--gamma', default=10.0, type=float)
    parser.add_argument('--epsilon', default=2.0, type=float)
    parser.add_argument('--hidden_dim', default=128, type=float)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--num_cpu', default=10, type=int)
    parser.add_argument('--adversarial_temperature', default=1.0, type=float)
    
    parser.add_argument('--seed', default=12345, type=int)

    args = parser.parse_args()
    args_str = json.dumps(vars(args))

    args.gpu = torch.device('cuda:' + args.gpu)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    init_dir(args)
    writer = SummaryWriter(os.path.join(args.tb_log_dir, args.name))
    args.writer = writer
    init_logger(args)
    logging.info(args_str)
    
    all_data = pickle.load(open(args.data_path, 'rb'))
    learner = FedE(args, all_data)
    learner.train()