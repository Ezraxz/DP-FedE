import json
import random
import os
from statistics import mean
import sys
import torch
import logging
import numpy as np


class Attacker_Client(object):
    def __init__(self, args):
        self.args = args
        
        #attack-1 && attack-2
        self.local_ent_embedding = None
        self.local_rel_embedding = None
        self.local_triples = None
        self.align_ent_list = None
        self.align_rel_list = None
        self.diff_ent_embed = None
        
        #attack-2
        self.test_2_triples = None
        self.triples_score_1 = []
        
        #evaulate
        self.len_test_tri = 0
        self.target_triples = None
    
        self.client_attack_res = {}
        self.client_attack_res["passive_transfer"] = {}
        self.client_attack_res["active_transfer"] = {}

    def attack_1(self, target_triples, rel_target2attack, align_rel_list):
        self.align_rel_list = align_rel_list
        test_triples = self.make_test_data(target_triples, rel_target2attack)
        self.test_2_triples = test_triples
        self.len_test_tri = len(test_triples)
        logging.info('Attack-1 result: ')
        self.client_attack_res["passive_transfer"]["threshold"] = []
        self.client_attack_res["passive_transfer"]["precision"] = []
        self.client_attack_res["passive_transfer"]["recall"] = []
        self.client_attack_res["passive_transfer"]["f1score"] = []
        for i in range(20):
            self.client_attack_res["passive_transfer"]["threshold"].append(i*0.02)
            evaulate_triples = self.compare_score(test_triples, i*0.02)
            self.evaulate(len(test_triples), evaulate_triples, rel_target2attack, 1)
        mean_f1_score = mean(self.client_attack_res["passive_transfer"]["f1score"])
        self.client_attack_res["passive_transfer"]["mean_f1_score"] = mean_f1_score
            
    def attack_2(self, updated_ent_embed, rel_target2attack):
        logging.info('Attack-2 result: ')
        self.client_attack_res["active_transfer"]["threshold"] = []
        self.client_attack_res["active_transfer"]["precision"] = []
        self.client_attack_res["active_transfer"]["recall"] = []
        self.client_attack_res["active_transfer"]["f1score"] = []
        for i in range(20):
            self.client_attack_res["active_transfer"]["threshold"].append(i*0.02)
            evaulate_triples = self.compute_s2(updated_ent_embed, i*0.02)
            self.evaulate(self.len_test_tri, evaulate_triples, rel_target2attack, 2)
        mean_f1_score = mean(self.client_attack_res["active_transfer"]["f1score"])
        self.client_attack_res["active_transfer"]["mean_f1_score"] = mean_f1_score
        json.dump(self.client_attack_res, open(self.args.attack_res_dir + '/' + self.args.name +'.json', 'w'))
        sys.exit()
    
    
    def get_local_data(self, local_ent_embed, local_rel_embed, local_triples):
        self.local_ent_embedding = local_ent_embed
        self.local_rel_embedding = local_rel_embed
        self.local_triples = local_triples.tolist()    
        
    def get_target_embedding(self, updated_ent_embed):
        align_ent_embed = torch.sub(self.local_ent_embedding, updated_ent_embed)
        align_ent_list = torch.sum(align_ent_embed, 1)
        self.align_ent_list = align_ent_list
        self.align_ent_list[align_ent_list != 0] = 1
        
        updated_ent_embed = updated_ent_embed * self.args.num_client
        diff_ent_embed = torch.sub(self.local_ent_embedding, updated_ent_embed)
        self.diff_ent_embed = torch.div(diff_ent_embed, self.args.num_client - 1)
         
    def make_test_data(self, target_triples, rel_target2attack):
        logging.info("Making test data...")
        test_triples = []
        
        rel_attack2target = {v: k for k, v in rel_target2attack.items()}
        self.target_triples = target_triples.tolist()
     
        for tri in self.target_triples:
            h, r, t = tri
            if self.align_ent_list[h] == 1 and self.align_ent_list[t] == 1 and r in rel_target2attack.keys():
                test_tri = [h, rel_target2attack[r], t]
                if test_tri not in self.local_triples:
                    test_triples.append(test_tri)
            if len(test_triples) >= self.args.test_data_count / 2:
                break
       
        num_true = len(test_triples)
        
        align_ent_id = []
        for id in range(len(self.align_ent_list)):
            if self.align_ent_list[id] == 1:
                align_ent_id.append(id)

        while len(test_triples) < 2 * num_true:
            h = random.choice(align_ent_id)
            t = random.choice(align_ent_id)
            while h == t:
                t = random.choice(align_ent_id)
            r = random.choice(self.align_rel_list)
            
            tri = [h, r, t]
            tri_1 = [h, rel_attack2target[r], t]
            if tri not in self.local_triples and tri_1 not in self.target_triples:
                test_triples.append(tri)

        return test_triples
            
    def compare_score(self, test_triples, add_thd):
        logging.info("Start comparing score")
        exist_tri = []
        for tri in test_triples:
            h, r, t = tri
            r = torch.tensor([r]).to(self.args.gpu)
            h = torch.tensor([h]).to(self.args.gpu)
            t = torch.tensor([t]).to(self.args.gpu)
            relation = torch.index_select(
                self.local_rel_embedding,
                dim=0,
                index=r
            ).unsqueeze(1)
            
            head = torch.index_select(
                self.local_ent_embedding,
                dim=0,
                index=h
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.local_ent_embedding,
                dim=0,
                index=t
            ).unsqueeze(1)
            pre_score = self.score_func(head, relation, tail)
            
            head = torch.index_select(
                self.diff_ent_embed,
                dim=0,
                index=h
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.diff_ent_embed,
                dim=0,
                index=t
            ).unsqueeze(1)
            rep_score = self.score_func(head, relation, tail)
            

            # if rep_score / pre_score > self.args.threshold_attack1 + add_thd or pre_score / rep_score > self.args.threshold_attack1 + add_thd:
            #     exist_tri.append(tri)
            
            if rep_score / pre_score  > self.args.threshold_attack1 + add_thd:
                exist_tri.append(tri)
            
            # if pre_score / rep_score > self.args.threshold_attack1 + add_thd:
            #     exist_tri.append(tri)
        
        return exist_tri
    
    def score_func(self, head, relation, tail):
        score = None
        if self.args.model == 'TransE':
            score = (head + relation) - tail
            score = torch.norm(score, p=1, dim=2)
        
        elif self.args.model == 'DistMult':
            score = (head * relation) * tail
            score = score.sum(dim = 2)
        
        return score

    def evaulate(self, len_test_data, triples, rel_target2attack, type):
        rel_attack2target = {v: k for k, v in rel_target2attack.items()}
        
        tp = 0
        fn = 0
        fp = 0
        tn = 0
       
        for tri in triples:
            h, r, t = tri
            tri_1 = [h, rel_attack2target[r], t]
            if tri_1 in self.target_triples:
                tp += 1
            else:
                fp += 1
        fn = len_test_data / 2 - tp
        tn = len_test_data / 2 - fp
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * precision * recall / (precision + recall)
        if type == 1:
            self.client_attack_res["passive_transfer"]["precision"].append(precision)
            self.client_attack_res["passive_transfer"]["recall"].append(recall)
            self.client_attack_res["passive_transfer"]["f1score"].append(f1_score)
        else:
            self.client_attack_res["active_transfer"]["precision"].append(precision)
            self.client_attack_res["active_transfer"]["recall"].append(recall)
            self.client_attack_res["active_transfer"]["f1score"].append(f1_score)
        
        logging.info('precision : {:.4f}, {} / {}'.format(precision, tp, tp + fp))
        logging.info('recall : {:.4f}, {} / {}'.format(recall, tp, tp + fn))
        logging.info('f1_score : {:.4f}'.format(f1_score))
            
    def reverse_tail(self, updated_ent_embed):
        logging.info("Reversing tail embedding...")
        reversed_ent = []
        updated_ent_embed_np = updated_ent_embed.cpu().detach().numpy()
       
        for tri in self.test_2_triples:
            h, r, t = tri
            if t in reversed_ent:
                continue
            reversed_ent.append(t)
            updated_ent_embed_np[t] = updated_ent_embed_np[t] * -1

        updated_ent_embed = torch.from_numpy(updated_ent_embed_np).to(self.args.gpu).requires_grad_()
        
        for tri in self.test_2_triples:
            h, r, t = tri
            r = torch.tensor([r]).to(self.args.gpu)
            h = torch.tensor([h]).to(self.args.gpu)
            t = torch.tensor([t]).to(self.args.gpu)
            relation = torch.index_select(
                self.local_rel_embedding,
                dim=0,
                index=r
            ).unsqueeze(1)
            
            head = torch.index_select(
                updated_ent_embed,
                dim=0,
                index=h
            ).unsqueeze(1)
            
            tail = torch.index_select(
                updated_ent_embed,
                dim=0,
                index=t
            ).unsqueeze(1)
            
            score = self.score_func(head, relation, tail)
            self.triples_score_1.append(score)
          
        return updated_ent_embed
    
    def compute_s2(self, updated_ent_embed, add_rhd):
        exist_tri = []
        for idx, tri in enumerate(self.test_2_triples):
            h, r, t = tri
            r = torch.tensor([r]).to(self.args.gpu)
            h = torch.tensor([h]).to(self.args.gpu)
            t = torch.tensor([t]).to(self.args.gpu)
            relation = torch.index_select(
                self.local_rel_embedding,
                dim=0,
                index=r
            ).unsqueeze(1)
            
            head = torch.index_select(
                updated_ent_embed,
                dim=0,
                index=h
            ).unsqueeze(1)
            
            tail = torch.index_select(
                updated_ent_embed,
                dim=0,
                index=t
            ).unsqueeze(1)
            
            score_1 = self.triples_score_1[idx]
            score_2 = self.score_func(head, relation, tail)

            if score_1 / score_2 > self.args.threshold_attack2 + add_rhd:
                exist_tri.append(tri)
               
            
        return exist_tri