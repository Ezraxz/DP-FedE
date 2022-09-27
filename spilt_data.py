import copy
from hashlib import new

import os
import random
import numpy as np
from collections import defaultdict as ddict
from tqdm import tqdm
import pickle

data_path = './dataset/FB15k-237'
data_size = 0.6
client_num = 2

ent2id_file = open(os.path.join(data_path, 'entity2id.txt'))
ent2id = dict()
num_ent = int(ent2id_file.readline())
for line in ent2id_file.readlines():
    ent, idx = line.split()
    ent2id[ent] = int(idx)
id2ent = {v: k for k, v in ent2id.items()}
    
rel2id = dict()
rel2id_file = open(os.path.join(data_path, 'relation2id.txt'))
num_rel = int(rel2id_file.readline())
for line in rel2id_file.readlines():
    rel, idx = line.split()
    rel2id[rel] = int(idx)
id2rel = {v: k for k, v in rel2id.items()}



train2id_file = open(os.path.join(data_path, 'train2id.txt'))
num_train = int(train2id_file.readline())
train_triples = []
for line in train2id_file.readlines():
    line = map(lambda x: int(x), line.split())
    h, t, r = line

    train_triples.append([h, r, t])

valid2id_file = open(os.path.join(data_path, 'valid2id.txt'))
num_valid = int(valid2id_file.readline())
valid_triples = []
for line in valid2id_file.readlines():
    line = map(lambda x: int(x), line.split())
    h, t, r = line

    valid_triples.append([h, r, t])

test2id_file = open(os.path.join(data_path, 'test2id.txt'))
num_test = int(test2id_file.readline())
test_triples = []
for line in test2id_file.readlines():
    line = map(lambda x: int(x), line.split())
    h, t, r = line

    test_triples.append([h, r, t])

train_triples = np.array(train_triples)
# valid_triples = np.array(valid_triples)
# test_triples = np.array(test_triples)

h_ent_pool = np.unique(train_triples[:,0])
t_ent_pool = np.unique(train_triples[:,2])
ent_list = np.unique(np.hstack((h_ent_pool, t_ent_pool))) 
ent_list_2 = ent_list.copy()

client_data = []

client_ent = []
train_triples = train_triples.tolist()

for i in tqdm(range(client_num)):
    newNumEnt = round(num_ent * data_size)
    
    num_client_avg = round(len(ent_list) / client_num)
    
    client_ent.append([])
    if i != client_num - 1:
        client_ent[i] = (np.random.choice(ent_list, num_client_avg, replace=False))
        ent_pool = np.setdiff1d(ent_list, client_ent, assume_unique=True)
    else:
        client_ent[i] = ent_list
    
    client_ent_list = client_ent[i].tolist()

    while len(client_ent_list) < newNumEnt:
        e = random.choice(ent_list_2)
        if e not in client_ent_list:
            client_ent_list.append(e)

    
    sampleEnt = []
    
    for e in client_ent_list:
        sampleEnt.append(id2ent[e])


    newEnt2id = dict()
    newRel2id = dict()

    entPool = []
    relPool = []

    trainTriples = []
    vaildTriples = []
    testTriples = []
    
    ent2num = dict()
    rel2num = dict()
    
    for key in sampleEnt: 
        entPool.append(ent2id[key])
        ent2num[ent2id[key]] = 0

    entPool1 = []

    print("choice train triples.")
    for tri in tqdm(train_triples):
        h, r, t = tri
        if h in entPool and t in entPool:
            relPool.append(r)
            if r in rel2num.keys():
                rel2num[r] += 1
            else:
                rel2num[r] = 1
            
            trainTriples.append(tri)
            ent2num[h] += 1
            ent2num[t] += 1
            entPool1.append(h)
            entPool1.append(t)
            
    len_ori = len(trainTriples)
    random.shuffle(trainTriples)        
    for tri in trainTriples:
        h, r, t = tri
        if ent2num[h] > 1 and ent2num[t] > 1 and rel2num[r] > 1:
            trainTriples.remove(tri)
            ent2num[h] -= 1
            ent2num[t] -= 1
            rel2num[r] -= 1
        if len(trainTriples) < len_ori * 0.75:
            break

    entPool = list(set(entPool1))
    for id in entPool:
        ent = id2ent[id]
        newEnt2id[ent] = ent2id[ent]
           

    
    print("choice valid triples.")
    for tri in tqdm(valid_triples):
        h, r, t = tri
        if h in entPool and t in entPool:
            relPool.append(r)
            vaildTriples.append(tri)

    print("choice test triples.")
    for tri in tqdm(test_triples):
        h, r, t = tri
        if h in entPool and t in entPool:
            relPool.append(r)
            testTriples.append(tri)

    relPool = list(set(relPool))
    
    for rel in relPool:
        relName = id2rel[rel]
        newRel2id[relName] = rel

    idx = 0
    for key in newEnt2id:
        newEnt2id[key] = idx
        idx += 1

    idx = 0

    for key in newRel2id:
        newRel2id[key] = idx
        idx += 1

    newTrainTriples = copy.deepcopy(trainTriples) 
    newValidTriples = copy.deepcopy(vaildTriples)
    newTestTriples = copy.deepcopy(testTriples)

    for i in range(len(trainTriples)):
        h, r, t = trainTriples[i]
        head = id2ent[h]
        tail = id2ent[t]
        relation = id2rel[r]
        newTrainTriples[i] = [newEnt2id[head], newRel2id[relation], newEnt2id[tail]]

    for i in range(len(vaildTriples)):
        h, r, t = vaildTriples[i]
        head = id2ent[h]
        tail = id2ent[t]
        relation = id2rel[r]
        newValidTriples[i] = [newEnt2id[head], newRel2id[relation], newEnt2id[tail]]

    for i in range(len(testTriples)):
        h, r, t = testTriples[i]
        head = id2ent[h]
        tail = id2ent[t]
        relation = id2rel[r]
        newTestTriples[i] = [newEnt2id[head], newRel2id[relation], newEnt2id[tail]]
        
    train_edge_index_ori = np.array(trainTriples)[:, [0, 2]].T
    train_edge_type_ori = np.array(trainTriples)[:, 1].T
    train_edge_index = np.array(newTrainTriples)[:, [0, 2]].T
    train_edge_type = np.array(newTrainTriples)[:, 1].T

    valid_edge_index_ori = np.array(vaildTriples)[:, [0, 2]].T
    valid_edge_type_ori = np.array(vaildTriples)[:, 1].T
    valid_edge_index = np.array(newValidTriples)[:, [0, 2]].T
    valid_edge_type = np.array(newValidTriples)[:, 1].T

    test_edge_index_ori = np.array(testTriples)[:, [0, 2]].T
    test_edge_type_ori = np.array(testTriples)[:, 1].T
    test_edge_index = np.array(newTestTriples)[:, [0, 2]].T
    test_edge_type = np.array(newTestTriples)[:, 1].T

    client_data_dict = {'train': {'edge_index': train_edge_index, 'edge_type': train_edge_type, 
                            'edge_index_ori': train_edge_index_ori, 'edge_type_ori': train_edge_type_ori},
                'test': {'edge_index': test_edge_index, 'edge_type': test_edge_type, 
                            'edge_index_ori': test_edge_index_ori, 'edge_type_ori': test_edge_type_ori},
                'valid': {'edge_index': valid_edge_index, 'edge_type': valid_edge_type, 
                        'edge_index_ori': valid_edge_index_ori, 'edge_type_ori': valid_edge_type_ori}}
    client_data.append(client_data_dict)

pickle.dump(client_data, open('./data/fb15k237-2-attack.pkl', 'wb'))
    
