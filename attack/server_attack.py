import random
import logging
import torch
import numpy as np
from sklearn.cluster import KMeans



class Attacker_Server(object):
    def __init__(self, args):
        self.args = args
        
        #attack-3
        self.test_links = None
        self.true_labels = dict()
        self.rel_num = 0
        self.center_links = []
        self.center_links_embed = []
        
        #attack-4
        self.test_links_ent_4 = None
        self.ent_embed_1 = None
        self.ent_embed_2 = None
        self.num_link = 0
        self.false_link_ent = []
        
        #evaulate
        self.target_triples = None
        self.target_links = []
        
        self.server_attack_res = {}
        self.server_attack_res["passive_relation"] = {}
        self.server_attack_res["active_relation"] = {}
        
    def get_eva_info(self, target_triples):
        self.target_triples = target_triples.tolist()
    
    def make_test_data(self):
        test_triples_index = np.random.choice(len(self.target_triples), round(self.args.test_data_count / 2)).tolist()
        test_triples = []
        test_links = []
        ent_index = []
        rel_index = []
        for idx in test_triples_index:
            h, r, t = self.target_triples[idx]
            if h == t:
                continue
            test_triples.append(self.target_triples[idx])
            ent_index.append(h)
            ent_index.append(t)
            if r not in rel_index:
                self.center_links.append([h, t])
            rel_index.append(r)
            self.target_links.append([h, 1, t])
            test_links.append([h, 1, t])
            
        ent_index = list(set(ent_index))
        rel_index = list(set(rel_index))
        self.rel_type = len(rel_index)
        self.rel_num = round(self.args.rel_num_multiple * len(test_triples))
        
        len_test = len(test_triples) * 2
                
        while len(test_triples) < len_test:
            h, t = random.sample(ent_index, 2)
            while h == t:
                h, t = random.sample(ent_index, 2)
            r = random.choice(rel_index)
            tri = [h, r, t]
            if tri not in test_triples and [h, 1, t] not in self.target_links:
                test_triples.append(tri)
                test_links.append([h, 1, t])
        
        random.shuffle(test_links)
        
        for idx, tri in enumerate(test_links):
            if tri in self.target_links:
                self.true_labels[idx] = 1
            else:
                self.true_labels[idx] = 0
        
        self.test_links = test_links
    

    def attack_3(self, target_ent_embed):
        link_embeds = self.get_links(target_ent_embed)
        link_label_score = self.clustering(link_embeds)
        sorted_links = self.sort_links(link_label_score)
        self.evaulate(sorted_links)
    
    def attack_4(self, updated_embed_2):
        sorted_ent = self.compute_influence(updated_embed_2)
        tp, fn, fp, tn = self.evaulate_4(sorted_ent)
        return tp, fn, fp, tn
        
    def get_prob_rel(self, head, tail):
        if self.args.model == 'TransE':
            link_embed = tail - head
        
        elif self.args.model == 'DistMult':
            link_embed = tail - head
        
        return link_embed
    
    def get_links(self, target_ent_embed):
        link_embeds = []
        test_links = self.test_links
        for tri in test_links:
            h, r, t = tri
            h = torch.tensor([h]).to(self.args.gpu)
            t = torch.tensor([t]).to(self.args.gpu)
            head = torch.index_select(
                target_ent_embed,
                dim=0,
                index=h
            ).unsqueeze(1)
            
            tail = torch.index_select(
                target_ent_embed,
                dim=0,
                index=t
            ).unsqueeze(1)
            
            link_embeds.append(self.get_prob_rel(head, tail).squeeze().cpu().numpy().tolist())
        
        for tri in self.center_links:
            h, t = tri
            h = torch.tensor([h]).to(self.args.gpu)
            t = torch.tensor([t]).to(self.args.gpu)
            head = torch.index_select(
                target_ent_embed,
                dim=0,
                index=h
            ).unsqueeze(1)
            
            tail = torch.index_select(
                target_ent_embed,
                dim=0,
                index=t
            ).unsqueeze(1)
            self.center_links_embed.append(self.get_prob_rel(head, tail).squeeze().cpu().numpy().tolist())
            
        return link_embeds
            
    def clustering(self, link_embeds):
        # print(link_embeds)
        link_label_score = []
        self.center_links_embed = np.array(self.center_links_embed)
        link_embeds = np.array(link_embeds)

        cluster_kmeans = KMeans(n_clusters=self.rel_type,n_init=self.rel_type, init=self.center_links_embed).fit(link_embeds)
        labels = cluster_kmeans.predict(link_embeds)
        transforms = cluster_kmeans.transform(link_embeds)
        logging.info("finish clustering")
        for i in range(len(link_embeds)):
            link_label_score.append([labels[i], transforms[i][labels[i]]])
        
        return link_label_score
    
    def sort_links(self, link_label_score):
        label2sorted = dict()
        id2sorted = dict()
        for idx, link in enumerate(self.test_links):
            label = link_label_score[idx][0]
            score = link_label_score[idx][1]
            if label not in label2sorted.keys():
                label2sorted[label] = []
                label2sorted[label].append([idx, score])
            else:
                label2sorted[label].append([idx, score])
            id2sorted[idx] = score
        
        for label in label2sorted.keys():
            sorted(label2sorted[label], key=(lambda x:x[1]))
        
        id2sorted = sorted(id2sorted.items(), key=lambda item:(item[1]))
        id2sorted = {k:v for k,v in id2sorted}
        
        sorted_links = []
        for idx in id2sorted.keys():
            sorted_links.append(idx)
            if len(sorted_links) >= self.rel_num:
                break
        return sorted_links

    def evaulate(self, sorted_links):
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        
        for idx in sorted_links:
            if self.true_labels[idx] == 1:
                tp += 1
            else:
                fp += 1
        
        fn = round(len(self.test_links) / 2) - tp
        tn = round(len(self.test_links) / 2) - fp
            
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * precision * recall / (precision + recall)
        
        self.server_attack_res["passive_relation"]["precision"] = precision
        self.server_attack_res["passive_relation"]["recall"] = recall
        self.server_attack_res["passive_relation"]["f1_score"] = f1_score
        
        logging.info('precision : {}, {} / {}'.format(precision, tp, tp + fp))
        logging.info('recall : {}, {} / {}'.format(recall, tp, tp + fn))   
        logging.info('f1 score : {}'.format(f1_score))
    
    def select_ent(self, ent_mask):
        self.ent_attack_4 = np.random.choice(len(ent_mask.cpu().numpy()))
        
        logging.info('select ent index: {}'.format(self.ent_attack_4))
        while ent_mask[self.ent_attack_4] == 0:
            self.ent_attack_4 = np.random.choice(len(ent_mask.cpu().numpy()))
        
        test_links = []
        for tri in self.target_triples:
            h, r, t = tri
            if h == self.ent_attack_4 or t == self.ent_attack_4:
                if h != self.ent_attack_4:
                    test_links.append(h)
                if t != self.ent_attack_4:
                    test_links.append(t)
        
        test_links = list(set(test_links))
        
        self.false_link_ent.clear()
        
        while len(self.false_link_ent) < len(test_links):
            ent = np.random.choice(len(ent_mask.cpu().numpy()))
            if ent_mask[ent] != 0 and ent not in test_links:
                self.false_link_ent.append(ent)        

        self.test_links_ent_4 = test_links
        self.num_link = len(test_links)         
    
    def add_noise(self, updated_embed):
        logging.info('Adding noise...')
        updated_embed_np = updated_embed.cpu().detach().numpy()
        noise = np.full(self.args.hidden_dim, 0.2)
        updated_embed_np[self.ent_attack_4] = updated_embed_np[self.ent_attack_4] + noise
        # updated_embed_np[self.ent_attack_4] = updated_embed_np[self.ent_attack_4] * 3
        self.len_test_4 = len(updated_embed_np)
        updated_embed = torch.from_numpy(updated_embed_np).to(self.args.gpu).requires_grad_()
        self.ent_embed_1 = updated_embed.cpu().clone().detach()

        return updated_embed

    def compute_influence(self, updated_embed):
        logging.info('Compute influence')
        self.ent_embed_2 = updated_embed.cpu().detach()
        influence_matrix = self.ent_embed_2 - self.ent_embed_1
        influence_score = torch.norm(influence_matrix, p=2, dim=1).numpy()
        
        id2score = dict()
        for idx, score in enumerate(influence_score):
             id2score[idx] = score
        
        id2score = sorted(id2score.items(), key=lambda item:(item[1]), reverse=True)
        id2score = {k:v for k,v in id2score}
        
        sorted_ent = []
        for idx in id2score.keys():
            if idx in self.false_link_ent or idx in self.test_links_ent_4:
                sorted_ent.append(idx)
                if len(sorted_ent) >= round(self.num_link * self.args.rel_num_multiple):
                    break
        
        # for idx, ent in enumerate(id2score.keys()):
        #     if ent in self.test_links_ent_4:
        #         print(idx)
        
        return sorted_ent
    
    def evaulate_4(self, sorted_ent):
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        for ent in sorted_ent:
            if ent in self.test_links_ent_4:
                tp += 1
            else:
                fp += 1
        
        fn = len(self.test_links_ent_4) - tp
        tn = self.len_test_4 - len(self.test_links_ent_4) - fp
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 0
        if precision != 0 or recall != 0:
            f1_score = 2 * precision * recall / (precision + recall)
                
        logging.info('precision : {:.4f}, {} / {}'.format(precision, tp, tp + fp))
        logging.info('recall : {:.4f}, {} / {}'.format(recall, tp, tp + fn))  
        logging.info('f1 score : {:.4f}'.format(f1_score))

        return tp, fn, fp, tn