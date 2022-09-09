

import operator
from functools import reduce
import random
import sys
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict as ddict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class KGBatchSampler:
    def __init__(self, dataset, batch_size, batch_list):
        self.length = len(dataset)
        self.batch_size = batch_size
        self.batch_list = batch_list
        
    def __iter__(self):
        for _ in range(round(self.length / self.batch_size)):
            indices = random.choice(self.batch_list)
            if len(indices) == self.batch_size:
                yield indices

    def __len__(self):
        return round(self.length / self.batch_size)

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, negative_sample_size):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.negative_sample_size = negative_sample_size
        
        self.hr2t = ddict(set)
        for h, r, t in triples:
            self.hr2t[(h, r)].add(t)
        for h, r in self.hr2t:
            self.hr2t[(h, r)] = np.array(list(self.hr2t[(h, r)]))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                self.hr2t[(head, relation)],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, idx

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        sample_idx = torch.tensor([_[2] for _ in data])
        return positive_sample, negative_sample, sample_idx


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, ent_mask=None):
        self.len = len(triples)
        self.triple_set = all_true_triples
        self.triples = triples
        self.nentity = nentity

        self.ent_mask = ent_mask

        self.hr2t_all = ddict(set)
        for h, r, t in all_true_triples:
            self.hr2t_all[(h, r)].add(t)

    def __len__(self):
        return self.len

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        return triple, trp_label

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        label = self.hr2t_all[(head, relation)]
        trp_label = self.get_label(label)
        triple = torch.LongTensor((head, relation, tail))

        return triple, trp_label

    def get_label(self, label):
        y = np.zeros([self.nentity], dtype=np.float32)

        if type(self.ent_mask) == np.ndarray:
            y[self.ent_mask] = 1.0

        for e2 in label:
            y[e2] = 1.0
        return torch.FloatTensor(y)


def get_all_clients(all_data, args):
    all_ent = np.array([], dtype=int)
    for data in all_data:
        all_ent = np.union1d(all_ent, data['train']['edge_index_ori'].reshape(-1))
    nentity = len(all_ent)

    train_dataloader_list = []
    test_dataloader_list = []
    valid_dataloader_list = []
    rel_embed_list = []

    ent_freq_list = []
    
    for data in tqdm(all_data):
        nrelation = len(np.unique(data['train']['edge_type']))

        train_triples = np.stack((data['train']['edge_index_ori'][0],
                                  data['train']['edge_type'],
                                  data['train']['edge_index_ori'][1])).T
        
        ent2tri = [[] for i in range(nentity)]
        # print("三元组总数: ", len(train_triples))
        for idx, tri in enumerate(train_triples):
            h, r, t = tri
            ent2tri[h].append(idx)
            ent2tri[t].append(idx)
            
            
        # sum = 0
        # num = 0
        # sorttri = {}
        # for item in ent2tri:
        #     if(len(item) > 0):
        #         if len(item) not in sorttri.keys():
        #             sorttri[len(item)] = 1
        #         else:
        #             sorttri[len(item)] += 1
        #         sum += len(item)
        #         num += 1
        # ord = sorted(sorttri.keys(), reverse=True)
        # ordtrinum = {}
        # for idx in ord:
        #     ordtrinum[idx] = sorttri[idx]
        # print("每个实体所属三元组个数: " , ordtrinum)
        # print("每个实体所属三元组平均个数: " ,sum / num)
        # sys.exit()
        
        ent2tri = reduce(operator.add, ent2tri)
        batch_list = []
        for i in range(round(len(ent2tri) / args.batch_size)):
            batch_list.append(ent2tri[i*args.batch_size : (i+1)*args.batch_size])
        

        valid_triples = np.stack((data['valid']['edge_index_ori'][0],
                                  data['valid']['edge_type'],
                                  data['valid']['edge_index_ori'][1])).T

        test_triples = np.stack((data['test']['edge_index_ori'][0],
                                 data['test']['edge_type'],
                                 data['test']['edge_index_ori'][1])).T

        client_mask_ent = np.setdiff1d(np.arange(nentity),
                                       np.unique(data['train']['edge_index_ori'].reshape(-1)), assume_unique=True)

        all_triples = np.concatenate([train_triples, valid_triples, test_triples])
        train_dataset = TrainDataset(train_triples, nentity, args.num_neg)
        valid_dataset = TestDataset(valid_triples, all_triples, nentity, client_mask_ent)
        test_dataset = TestDataset(test_triples, all_triples, nentity, client_mask_ent)

        # dataloader
        train_dataloader = DataLoader(
            train_dataset,
            # batch_size=args.batch_size,
            batch_sampler=KGBatchSampler(train_dataset, args.batch_size, batch_list),
            collate_fn=TrainDataset.collate_fn,
        )
        train_dataloader_list.append(train_dataloader)

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=args.test_batch_size,
            collate_fn=TestDataset.collate_fn
        )
        valid_dataloader_list.append(valid_dataloader)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            collate_fn=TestDataset.collate_fn
        )
        test_dataloader_list.append(test_dataloader)

        embedding_range = torch.Tensor([(args.gamma + args.epsilon) / args.hidden_dim])
        if args.model in ['ComplEx']:
            rel_embed = torch.zeros(nrelation, args.hidden_dim * 2).to(args.gpu).requires_grad_()
        else:
            rel_embed = torch.zeros(nrelation, args.hidden_dim).to(args.gpu).requires_grad_()

        nn.init.uniform_(
            tensor=rel_embed,
            a=-embedding_range.item(),
            b=embedding_range.item()
        )
        rel_embed_list.append(rel_embed)

        ent_freq = torch.zeros(nentity)
        for e in data['train']['edge_index_ori'].reshape(-1):
            ent_freq[e] += 1
        ent_freq_list.append(ent_freq)

    ent_freq_mat = torch.stack(ent_freq_list).to(args.gpu)

    return train_dataloader_list, valid_dataloader_list, test_dataloader_list, \
           ent_freq_mat, rel_embed_list, nentity
