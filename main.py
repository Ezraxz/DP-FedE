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
    parser.add_argument('--data_path', default='./data/fb15k237-2-attack.pkl', type=str)
    
    # parser.add_argument('--name', default='fed3_batch_64_noise_0.5_clip_1.2', type=str)
    # parser.add_argument('--name', default='fed2_server_attack_dp', type=str)
    # parser.add_argument('--name', default='fed2_client_attack_dp_250', type=str)
    parser.add_argument('--name', default='fed2', type=str)
    # parser.add_argument('--name', default='test', type=str)
    parser.add_argument('--state_dir', '-state_dir', default='./state/220920/fed2', type=str)
    parser.add_argument('--log_dir', '-log_dir', default='./log/220920', type=str)
    parser.add_argument('--tb_log_dir', '-tb_log_dir', default='./tb_log', type=str)
    parser.add_argument('--attack_embed_dir', '-attack_embed_dir', default='./state/220920/fed2/fed2.', type=str)
    parser.add_argument('--attack_res_dir', '-attack_res_dir', default='./attack_res/220920', type=str)
    
    parser.add_argument('--model', default='TransE', choices=['TransE', 'RotatE', 'DistMult', 'ComplEx'])
    
    # one task hyperparam
    parser.add_argument('--one_client_idx', default=0, type=int)
    parser.add_argument('--max_epoch', default=10000, type=int)
    parser.add_argument('--log_per_epoch', default=1, type=int)
    parser.add_argument('--check_per_epoch', default=10, type=int)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int)
    parser.add_argument('--num_neg', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=int)
    
    # for FedE
    parser.add_argument('--num_client', default=2, type=int)
    parser.add_argument('--max_round', default=1000, type=int)
    parser.add_argument('--local_epoch', default=1)
    parser.add_argument('--fraction', default=1, type=float)
    parser.add_argument('--log_per_round', default=1, type=int)
    parser.add_argument('--check_per_round', default=5, type=int)

    parser.add_argument('--early_stop_patience', default=5, type=int)
    parser.add_argument('--gamma', default=10.0, type=float)
    parser.add_argument('--epsilon', default=2.0, type=float)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--gpu', default='3', type=str)
    parser.add_argument('--num_cpu', default=10, type=int)
    parser.add_argument('--adversarial_temperature', default=1.0, type=float)
    
    #defense_param
    parser.add_argument('--use_dp', default=False, type=bool)
    parser.add_argument('--microbatch_size', default=1, type=int)
    parser.add_argument('--l2_norm_clip', default=1.2, type=float)
    parser.add_argument('--noise_multiplier', default=0.5, type=float)
    parser.add_argument('--sgd_eps', default=200.0, type=float)
    parser.add_argument('--svt_eps', default=8.0, type=float)
    
    #attack_param
    parser.add_argument('--is_attack', default=False, type=bool)
    parser.add_argument('--attack_type', default='client', type=str, choices=['server', 'client'])
    parser.add_argument('--target_client', default=0, type=int)
    parser.add_argument('--attack_client', default=1, type=int)
    parser.add_argument('--test_data_count', default=2000, type=int)
    parser.add_argument('--start_round', default=200, type=int)
    
    #attack-1_param
    parser.add_argument('--threshold_attack1', default=1.00, type=float)
    
    #attack-2_param
    parser.add_argument('--threshold_attack2', default=1.00, type=float)
    parser.add_argument('--cmp_round', default=5, type=int)
    
    #attack-3_param
    parser.add_argument('--rel_num_multiple', default=1.0, type=float)
    
    parser.add_argument('--seed', default=12345, type=int)

    args = parser.parse_args()
    args_str = json.dumps(vars(args))

    args.gpu = torch.device('cuda:' + args.gpu)
    if not args.is_attack:
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