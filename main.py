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
from fusion import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/fb13-2.pkl', type=str)
     
    parser.add_argument('--name', default='fed2_fb13_transe_dp_naive', type=str)
    parser.add_argument('--state_dir', '-state_dir', default='./state/fed2_fb13_transe_dp_naive', type=str)
    parser.add_argument('--log_dir', '-log_dir', default='./log/fed2_fb13_transe_dp_naive', type=str)
    parser.add_argument('--tb_log_dir', '-tb_log_dir', default='./tb_log', type=str)
    parser.add_argument('--attack_embed_dir', '-attack_embed_dir', 
                            default='./state/fed4_fb15k237_transe/fed4_fb15k237_transe.', type=str)
    parser.add_argument('--attack_res_dir', '-attack_res_dir', default='./attack_res/', type=str)
    
    parser.add_argument('--model', default='TransE', choices=['TransE', 'RotatE', 'DistMult', 'ComplEx'])
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int)
    parser.add_argument('--num_neg', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=int)
    
    parser.add_argument('--model_mode', default='fusion', type=str, choices=['fusion', 'nofusion'])
    
    # for FedE
    parser.add_argument('--num_client', default=2, type=int)
    parser.add_argument('--max_round', default=1000, type=int)
    parser.add_argument('--local_epoch', default=1)
    parser.add_argument('--fraction', default=1, type=float)
    parser.add_argument('--log_per_round', default=1, type=int)
    parser.add_argument('--check_per_round', default=5, type=int)

    parser.add_argument('--early_stop_patience', default=100, type=int)
    parser.add_argument('--gamma', default=10.0, type=float)
    parser.add_argument('--epsilon', default=2.0, type=float)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--num_cpu', default=10, type=int)
    parser.add_argument('--adversarial_temperature', default=1.0, type=float)
    
    #defense_param
    parser.add_argument('--use_dp', default=True, type=bool)
    parser.add_argument('--naive', default=True, type=bool)
    parser.add_argument('--microbatch_size', default=1, type=int)
    parser.add_argument('--l2_norm_clip', default=1.2, type=float)
    parser.add_argument('--noise_multiplier', default=0.4, type=float)
    parser.add_argument('--sgd_eps', default=70.0, type=float)
    parser.add_argument('--topk_eps', default=2.0, type=float)
    parser.add_argument('--diff_mrr', default=0.0010, type=float)
    parser.add_argument('--decline_mult', default=1, type=float)
    
    #attack_param
    parser.add_argument('--is_attack', default=False, type=bool)
    parser.add_argument('--attack_type', default='client', type=str, choices=['server', 'client', 'collusion', 'make_data'])
    parser.add_argument('--target_client', default=0, type=int)
    parser.add_argument('--attack_client', default=1, type=int)
    parser.add_argument('--test_data_count', default=1000, type=int)
    parser.add_argument('--start_round', default=0, type=int)
    
    #attack-1_param
    parser.add_argument('--threshold_attack1', default=1.0, type=float)
    
    #attack-2_param
    parser.add_argument('--threshold_attack2', default=1.0, type=float)
    parser.add_argument('--cmp_round', default=1, type=int)
    
    #attack-3_param
    parser.add_argument('--rel_num_multiple', default=0.5, type=float)
    
    #test
    parser.add_argument('--test_mode', default='fake', type=str, choices=['normal', 'fake'])
    
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
    
    if args.model_mode == 'fusion':
        train_fusion(args, all_data, 3, 'nell995_fed3_client_{}_rotate.ckpt', 'nell995_fed3_fed_rotate.ckpt')
    
    else:
        if args.is_attack != True:
            learner = FedE(args, all_data)
            learner.train()
        else:
            learner = FedE(args, all_data)
            # learner.train() #make attack data
            if args.use_dp != True:
                for i in range(20):
                    round = (i+1) * 5
                    if os.path.exists(args.attack_embed_dir + str(round) + '.ckpt'):
                        learner.before_attack_load(round)
                        learner.train(round)
            else:
                rounds = [2,4,8,16,24,32,48,64]
                for round in rounds:
                    if os.path.exists(args.attack_embed_dir + 'eps_' + str(round) + '.ckpt'):
                        learner.before_attack_load(round)
                        learner.train(round)
            