import torch
from torch.optim import Optimizer
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions.normal import Normal
from torch.optim import SGD, Adam, Adagrad, RMSprop
import sys
import numpy as np

def make_optimizer_class(cls):
    class DPOptimizerClass(cls):
        def __init__(self, margs, len_embed, l2_norm_clip, noise_multiplier, minibatch_size, microbatch_size, *args, **kwargs):
            super(DPOptimizerClass, self).__init__(*args, **kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size
            # self.svt_eps = margs.svt_eps
            # self.gpu = margs.gpu
            
            # self.t_noise = (self.l2_norm_clip / 1.5 + np.random.laplace(0, self.l2_norm_clip / self.svt_eps, 1))[0]
            # self.norm_noise = torch.from_numpy(np.random.laplace(0, self.l2_norm_clip / self.svt_eps, len_embed)).to(self.gpu) 

            self.param_groups[0]['embed_type'] = 'relation'
            self.param_groups[1]['embed_type'] = 'entity'
            for group in self.param_groups:
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]
                group['dp_accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]

        def zero_microbatch_grad(self):
            super(DPOptimizerClass, self).zero_grad()

        def microbatch_step(self, sample_type=True):
            if sample_type == True:
                total_norm = 0.
                for group in self.param_groups:
                    for param in group['params']:
                        if param.requires_grad:
                            if group['embed_type'] == 'entity':
                                total_norm += param.grad.data.norm(2).item() ** 2.
                                
                total_norm = total_norm ** .5
                clip_coef = min(self.l2_norm_clip / (total_norm + 1e-6), 1.)
       
            for group in self.param_groups:
                for param, accum_grad, dp_accum_grad in zip(group['params'], group['accum_grads'], group['dp_accum_grads']):
                    if param.requires_grad:
                        if group['embed_type'] == 'entity' and sample_type == True:
                            dp_accum_grad.add_(param.grad.data.mul(clip_coef))
                        else:
                            accum_grad.add_(param.grad.data)

        def zero_grad(self):
            for group in self.param_groups:
                for accum_grad, dp_accum_grad in zip(group['accum_grads'], group['dp_accum_grads']):
                    if accum_grad is not None:
                        accum_grad.zero_()
                    if dp_accum_grad is not None:
                        dp_accum_grad.zero_()

        def step(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad, dp_accum_grad in zip(group['params'], group['accum_grads'], group['dp_accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()
                        if group['embed_type'] == 'entity':
                            param_w = torch.sum(dp_accum_grad, dim=1)
                            param_select = param_w
                            param_select[param_w != 0] = 1
                            param_select = param_select.unsqueeze(1)
                                                    
                            dp_accum_grad.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(dp_accum_grad))
                            dp_accum_grad.mul_(param_select)              
                            param.grad.data.add_(dp_accum_grad)
                            
                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)
            super(DPOptimizerClass, self).step(*args, **kwargs)

    return DPOptimizerClass

DPAdam = make_optimizer_class(Adam)
DPAdagrad = make_optimizer_class(Adagrad)
DPSGD = make_optimizer_class(SGD)
DPRMSprop = make_optimizer_class(RMSprop)

