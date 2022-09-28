

import sys
from dataloader import *
import json
import os
import copy
import logging
from kge_model import KGEModel
import torch.nn.functional as F

from pyvacy import optim, analysis, sampling
from attack import client_attack, server_attack
from utils import ThreadWithReturnValue

class Server(object):
    def __init__(self, args, nentity):
        self.args = args
        embedding_range = torch.Tensor([(args.gamma + args.epsilon) / args.hidden_dim])
        if args.model in ['RotatE', 'ComplEx']:
            self.ent_embed = torch.zeros(nentity, args.hidden_dim*2).to(args.gpu).requires_grad_()
        else:
            self.ent_embed = torch.zeros(nentity, args.hidden_dim).to(args.gpu).requires_grad_()
        nn.init.uniform_(
            tensor=self.ent_embed,
            a=-embedding_range.item(),
            b=embedding_range.item()
        )
        self.nentity = nentity
        self.before_emd = []

    def send_emb(self):
        self.before_emd = self.ent_embed
        return copy.deepcopy(self.ent_embed)


    def aggregation(self, clients, ent_update_weights, num_round):
        agg_ent_mask = ent_update_weights
        agg_ent_mask[ent_update_weights != 0] = 1
        ent_w_sum = torch.sum(agg_ent_mask, dim=0)
        ent_w = agg_ent_mask / ent_w_sum
        ent_w[torch.isnan(ent_w)] = 0
       
        if self.args.model in ['RotatE', 'ComplEx']:
            update_ent_embed = torch.zeros(self.nentity, self.args.hidden_dim * 2).to(self.args.gpu)
        else:
            update_ent_embed = torch.zeros(self.nentity, self.args.hidden_dim).to(self.args.gpu)
        for i, client in enumerate(clients):
            local_ent_embed = client.ent_embed.clone().detach()
            update_ent_embed += local_ent_embed * ent_w[i].reshape(-1, 1)
        self.ent_embed = update_ent_embed.requires_grad_()

    
class Client(object):
    def __init__(self, args, client_id, data, train_dataloader,
                 valid_dataloader, test_dataloader, rel_embed):
        self.args = args
        self.data = data
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.rel_embed = rel_embed
        self.client_id = client_id

        self.score_local = []
        self.score_global = []

        self.kge_model = KGEModel(args, args.model)
        self.ent_embed = None
        
        self.iteration = 0
        self.eps = 0
        self.loss = None
        
        self.real_parm = optim.real_parm(args, args.nentity)

    def __len__(self):
        return len(self.train_dataloader.dataset)

    def client_update(self):
        eps = 0
        if self.args.use_dp == True:
            if self.eps >= self.args.sgd_eps:
                return self.loss,  self.eps
            optimizer_dp = optim.DPAdam(
                margs=self.args,
                real_parm=self.real_parm,
                l2_norm_clip=self.args.l2_norm_clip,
                noise_multiplier= self.args.noise_multiplier,
                minibatch_size=self.args.batch_size,
                microbatch_size=self.args.microbatch_size, 
                params=[{'params': self.rel_embed}, {'params': self.ent_embed}],
                lr=self.args.lr
            )             

            losses = []
            eps_pre = analysis.epsilon(
                        len(self.train_dataloader.dataset),
                        self.args.batch_size,
                        self.args.noise_multiplier,
                        self.iteration,
                        1e-5
                    )
        
            for i in range(self.args.local_epoch):
                for batch in self.train_dataloader:
                    optimizer_dp.zero_grad()
                    positive_samples, negative_samples, sample_idx = batch
                    self.iteration += 1
                    for idx in range(self.args.batch_size):
                        idx = torch.tensor([idx]) 
                        positive_sample = torch.index_select(
                            positive_samples,
                            dim=0,
                            index=idx
                        )
                        negative_sample = torch.index_select(
                            negative_samples,
                            dim=0,
                            index=idx
                        )
                       
                        positive_sample = positive_sample.to(self.args.gpu)
                        negative_sample = negative_sample.to(self.args.gpu)

                        negative_score = self.kge_model((positive_sample, negative_sample),
                                                        self.rel_embed, self.ent_embed)

                        negative_score = (F.softmax(negative_score * self.args.adversarial_temperature, dim=1).detach()
                                        * F.logsigmoid(-negative_score)).sum(dim=1)

                        positive_score = self.kge_model(positive_sample,
                                                        self.rel_embed, self.ent_embed, neg=False)

                        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

                        positive_sample_loss = - positive_score.mean()
                        negative_sample_loss = - negative_score.mean()

                        loss = (positive_sample_loss + negative_sample_loss) / 2

                        positive_sample_loss = positive_sample_loss / 2
                        negative_sample_loss = negative_sample_loss / 2
                        
                        optimizer_dp.zero_microbatch_grad()
                        positive_sample_loss.backward()
                        optimizer_dp.microbatch_step(sample_type=True)
                        
                        optimizer_dp.zero_microbatch_grad()
                        negative_sample_loss.backward()
                        optimizer_dp.microbatch_step(sample_type=False)

                    optimizer_dp.step()
                    losses.append(loss.item())
                
                eps_train = analysis.epsilon(
                        len(self.train_dataloader.dataset),
                        self.args.batch_size,
                        self.args.noise_multiplier,
                        self.iteration,
                        1e-5
                    )
                
                eps = eps_train - eps_pre + self.eps
                self.eps = eps
                logging.info('Client {} : Achieves ({}, {})-DP'.format(
                    self.client_id,
                    eps,
                1e-5,
                ))
        
        if self.args.use_dp == False:
            optimizer = torch.optim.Adam([{'params': self.rel_embed},
                                    {'params': self.ent_embed}], lr=self.args.lr)

            losses = []
            
            for i in range(self.args.local_epoch):
                for batch in self.train_dataloader:
                    positive_sample, negative_sample, sample_idx = batch

                    positive_sample = positive_sample.to(self.args.gpu)
                    negative_sample = negative_sample.to(self.args.gpu)

                    negative_score = self.kge_model((positive_sample, negative_sample),
                                                    self.rel_embed, self.ent_embed)

                    negative_score = (F.softmax(negative_score * self.args.adversarial_temperature, dim=1).detach()
                                    * F.logsigmoid(-negative_score)).sum(dim=1)

                    positive_score = self.kge_model(positive_sample,
                                                    self.rel_embed, self.ent_embed, neg=False)

                    positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

                    positive_sample_loss = - positive_score.mean()
                    negative_sample_loss = - negative_score.mean()

                    loss = (positive_sample_loss + negative_sample_loss) / 2

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()                            

                    losses.append(loss.item())

        self.loss = np.mean(losses)
        return self.loss, eps
    


    def client_eval(self, istest=False):
        if istest:
            dataloader = self.test_dataloader
        else:
            dataloader = self.valid_dataloader

        results = ddict(float)
        for batch in dataloader:
            triplets, labels = batch
            triplets, labels = triplets.to(self.args.gpu), labels.to(self.args.gpu)
            head_idx, rel_idx, tail_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            pred = self.kge_model((triplets, None),
                                   self.rel_embed, self.ent_embed)
            b_range = torch.arange(pred.size()[0], device=self.args.gpu)
            target_pred = pred[b_range, tail_idx]
            pred = torch.where(labels.byte(), -torch.ones_like(pred) * 10000000, pred)
            pred[b_range, tail_idx] = target_pred

            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                      dim=1, descending=False)[b_range, tail_idx]

            ranks = ranks.float()
            count = torch.numel(ranks)

            results['count'] += count
            results['mr'] += torch.sum(ranks).item()
            results['mrr'] += torch.sum(1.0 / ranks).item()

            for k in [1, 5, 10]:
                results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

        for k, v in results.items():
            if k != 'count':
                results[k] /= results['count']

        return results
    


class FedE(object):
    def __init__(self, args, all_data):
        self.args = args

        train_dataloader_list, valid_dataloader_list, test_dataloader_list, \
            self.ent_freq_mat, rel_embed_list, nentity, self.all_train_triples = get_all_clients(all_data, args)

        self.args.nentity = nentity

        # client
        self.num_clients = len(train_dataloader_list)
        self.clients = [
            Client(self.args, i, all_data[i], train_dataloader_list[i], valid_dataloader_list[i],
                   test_dataloader_list[i], rel_embed_list[i]) for i in range(self.num_clients)
        ]

        self.server = Server(args, nentity)

        self.total_test_data_size = sum([len(client.test_dataloader.dataset) for client in self.clients])
        self.test_eval_weights = [len(client.test_dataloader.dataset) / self.total_test_data_size for client in self.clients]

        self.total_valid_data_size = sum([len(client.valid_dataloader.dataset) for client in self.clients])
        self.valid_eval_weights = [len(client.valid_dataloader.dataset) / self.total_valid_data_size for client in self.clients]

        if self.args.is_attack:
            if self.args.start_round != 0:
                self.before_attack_load()
            #transfer inference attack
            self.align_rel_list, self.rel_target2attack = get_attack_data(all_data, args)
            self.attacker_client = client_attack.Attacker_Client(args) 
            self.reverse_embed = None
            self.attack_idx = self.args.attack_client
            self.target_idx = self.args.target_client
            self.agg_ent_mask = self.ent_freq_mat
            self.agg_ent_mask[self.ent_freq_mat != 0] = 1
            
            #relation inference attack
            self.attacker_server = server_attack.Attacker_Server(args)
            self.attacker_server.get_eva_info(self.all_train_triples[self.args.target_client])
            # self.attacker_server.make_test_data()
            self.tp = 0
            self.fn = 0
            self.fp = 0
            self.tn = 0
        
        
    def write_training_loss(self, loss, e):
        self.args.writer.add_scalar("training/loss", loss, e)

    def write_evaluation_result(self, results, e):
        self.args.writer.add_scalar("evaluation/mrr", results['mrr'], e)
        self.args.writer.add_scalar("evaluation/hits10", results['hits@10'], e)
        self.args.writer.add_scalar("evaluation/hits5", results['hits@5'], e)
        self.args.writer.add_scalar("evaluation/hits1", results['hits@1'], e)

    def save_checkpoint(self, e):
        state = {'ent_embed': self.server.ent_embed,
                 'rel_embed': [client.rel_embed for client in self.clients]}
        # delete previous checkpoint
        # for filename in os.listdir(self.args.state_dir):
        #     if self.args.name in filename.split('.') and os.path.isfile(os.path.join(self.args.state_dir, filename)):
        #         os.remove(os.path.join(self.args.state_dir, filename))
        # save current checkpoint
        torch.save(state, os.path.join(self.args.state_dir,
                                       self.args.name + '.' + str(e) + '.ckpt'))

    def save_model(self, best_epoch):
        os.rename(os.path.join(self.args.state_dir, self.args.name + '.' + str(best_epoch) + '.ckpt'),
                  os.path.join(self.args.state_dir, self.args.name + '.best'))

    def adaptive_eps(self, k, client_res, pre_client_res):
        if client_res[k]['mrr'] - pre_client_res[k] < self.args.diff_mrr:
            self.clients[k].args.noise_multiplier *= self.args.decline_mult
            logging.info("client {} noise_multiplier {}".format(k, self.clients[k].args.noise_multiplier))
        pre_client_res[k] = client_res[k]['mrr']
    
    def send_emb(self):
        for k, client in enumerate(self.clients):
            client.ent_embed = self.server.send_emb()

    def train(self):
        best_epoch = 0
        best_mrr = 0
        bad_count = 0
        pre_client_res = [0] * self.num_clients
        visit = [2,4,8,16,24,32,48,64]
        if len(visit) > 0 and self.clients[self.target_idx].eps >= visit[0]:
            state = {'ent_embed': self.server.ent_embed,
                 'rel_embed': [client.rel_embed for client in self.clients]}
            torch.save(state, os.path.join(self.args.state_dir,
                                       self.args.name + '.eps_' + str(visit[0]) + '.ckpt'))
            visit.pop(0)
        
        for num_round in range(self.args.max_round):
            n_sample = max(round(self.args.fraction * self.num_clients), 1)
            sample_set = np.random.choice(self.num_clients, n_sample, replace=False)

            self.send_emb()
            
            #active transfer inference attack
            if self.args.is_attack and self.args.attack_type == 'client' and num_round == 1:
                logging.info('Attack-2 : attacker {}, target {}'.format(self.attack_idx, self.target_idx))
                self.reverse_embed = self.attacker_client.reverse_tail(self.clients[self.attack_idx].ent_embed)
            
            #fkg client train
            round_loss = 0
            eps_avg = 0
            threads = []
            for k in iter(sample_set):
                t = ThreadWithReturnValue(target=self.clients[k].client_update)
                t.start()
                threads.append(t)
            for t in threads:
                client_loss, eps = t.join()
                round_loss += client_loss
                eps_avg += eps
            round_loss /= n_sample
            eps_avg /= self.num_clients
            
            #active transfer inference attack
            if self.args.is_attack and self.args.attack_type == 'client'and num_round == 1:
                self.clients[self.attack_idx].ent_embed = self.reverse_embed
            
            #passive relation inference attack
            if self.args.is_attack and num_round == 0 and self.args.attack_type == 'server':
                logging.info('Attack-3 : attacker Server, target {}'.format(self.target_idx))
                self.attacker_server.attack_3(self.clients[self.target_idx].ent_embed.clone().detach())
            
            """
            #active relation inference attack
            if self.args.is_attack and num_round == 0 and self.args.attack_type == 'server':
                logging.info('Attack-4 : attacker Server, target {}'.format(self.target_idx))
                for k in iter(sample_set):
                    self.clients[k].args.local_epoch = 5
                self.attacker_server.select_ent(self.agg_ent_mask[self.args.target_client])
                self.input_ent_embed = self.clients[self.target_idx].ent_embed
                noise_embed = self.attacker_server.add_noise(self.input_ent_embed)
                self.clients[self.target_idx].ent_embed = noise_embed
                continue
            if self.args.is_attack and num_round >= 1 and num_round <= 10 and self.args.attack_type == 'server':
                tp, fn, fp, tn = self.attacker_server.attack_4(self.clients[self.target_idx].ent_embed)
                self.tp += tp
                self.fn += fn
                self.fp += fp
                self.tn += tn
                self.attacker_server.select_ent(self.agg_ent_mask[self.args.target_client])
                noise_embed = self.attacker_server.add_noise(self.input_ent_embed)
                self.clients[self.target_idx].ent_embed = noise_embed
                continue
            if self.args.is_attack and self.args.attack_type == 'server' and  num_round > 10:
                precision = self.tp / (self.tp + self.fp)
                recall = self.tp / (self.tp +  self.fn)
                f1_score = 2 * precision * recall / (precision + recall)
                logging.info('precision : {}, {} / {}'.format(precision, self.tp, self.tp + self.fp))
                logging.info('recall : {}, {} / {}'.format(recall, self.tp, self.tp +  self.fn))
                logging.info('f1 score : {}'.format(f1_score))
                
                self.attacker_server.server_attack_res["active_relation"]["precision"] = precision
                self.attacker_server.server_attack_res["active_relation"]["recall"] = recall
                self.attacker_server.server_attack_res["active_relation"]["f1_score"] = f1_score
                json.dump(self.attacker_server.server_attack_res, open(self.args.attack_res_dir + '/' + self.args.name +'.json', 'w'))
                sys.exit()  
            """   
                     
            #fkg server aggregation
            self.server.aggregation(self.clients, self.ent_freq_mat, num_round)
            
            #active transfer inference attack
            if self.args.is_attack and num_round == 1 + self.args.cmp_round and self.args.attack_type == 'client':
                self.attacker_client.attack_2(self.server.send_emb(), self.rel_target2attack)
            
            #passive transfer inference attack
            if self.args.is_attack and num_round == 0 and self.args.attack_type == 'client':
                logging.info('Attack-1 : attacker {}, target {}'.format(self.attack_idx, self.target_idx))
                self.attacker_client.get_local_data(self.clients[self.attack_idx].ent_embed, self.clients[self.attack_idx].rel_embed,
                                                    self.all_train_triples[self.attack_idx])
                self.attacker_client.get_target_embedding(self.server.send_emb())
                self.attacker_client.attack_1(self.all_train_triples[self.target_idx], self.rel_target2attack, 
                                              self.align_rel_list)
                        

            logging.info('round: {} | loss: {:.4f}'.format(num_round, np.mean(round_loss)))
            self.write_training_loss(np.mean(round_loss), num_round)

            if num_round % self.args.check_per_round == 0 and num_round != 0:
                eval_res, client_res = self.evaluate()
                self.write_evaluation_result(eval_res, num_round)
                for k in range(self.num_clients):
                    self.adaptive_eps(k, client_res, pre_client_res)
                if eval_res['mrr'] > best_mrr:
                    best_mrr = eval_res['mrr']
                    best_epoch = num_round
                    logging.info('best model | mrr {:.4f}'.format(best_mrr))
                    bad_count = 0
                else:
                    bad_count += 1
                    logging.info('best model is at round {0}, mrr {1:.4f}, bad count {2}'.format(
                        best_epoch, best_mrr, bad_count))
                self.save_checkpoint(num_round)

            if bad_count >= self.args.early_stop_patience or eps_avg >= self.args.sgd_eps:
                logging.info('early stop at round {}'.format(num_round))
                break

        logging.info('finish training')
        logging.info('save best model')
        self.save_model(best_epoch)
        self.before_test_load()
        self.evaluate(istest=True)

    def before_attack_load(self):
        logging.info("loading embedding")
        state = torch.load(self.args.attack_embed_dir + str(self.args.start_round) + '.ckpt', map_location=self.args.gpu)
        self.server.ent_embed = state['ent_embed']
        for idx, client in enumerate(self.clients):
            client.rel_embed = state['rel_embed'][idx]
        self.evaluate()
    
    def before_test_load(self):
        state = torch.load(os.path.join(self.args.state_dir, self.args.name + '.best'), map_location=self.args.gpu)
        self.server.ent_embed = state['ent_embed']
        for idx, client in enumerate(self.clients):
            client.rel_embed = state['rel_embed'][idx]

    def evaluate(self, istest=False):
        self.send_emb()
        result = ddict(int)
        if istest:
            weights = self.test_eval_weights
        else:
            weights = self.valid_eval_weights
        client_res_list = []
        for idx, client in enumerate(self.clients):
            client_res = client.client_eval(istest)
            client_res_list.append(client_res)  
            logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
                client_res['mrr'], client_res['hits@1'],
                client_res['hits@5'], client_res['hits@10']))

            for k, v in client_res.items():
                result[k] += v * weights[idx]

        logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
                     result['mrr'], result['hits@1'],
                     result['hits@5'], result['hits@10']))

        return result, client_res_list





