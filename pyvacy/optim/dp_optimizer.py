import torch
from torch.optim import Optimizer
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions.normal import Normal
from torch.optim import SGD, Adam, Adagrad, RMSprop
import sys
import numpy as np

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def real_parm(args, len):
    parm = []
    for idx in range(len-1):
        if idx >= 0.5 * args.batch_size and idx <= 2 * args.batch_size:
            p = args.l2_norm_clip / 50
        else:
            p = np.iinfo(np.int16).min
        parm.append(p)
    return torch.Tensor(parm).unsqueeze(1).to(args.gpu)

def make_optimizer_class(cls):
    class DPOptimizerClass(cls):
        def __init__(self, margs, real_parm, l2_norm_clip, noise_multiplier, minibatch_size, microbatch_size, *args, **kwargs):
            super(DPOptimizerClass, self).__init__(*args, **kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size
            self.args = margs
            self.real_parm = real_parm 

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
                            grad_norm = torch.norm(dp_accum_grad, p=2, dim=1).unsqueeze(1)
                            sorted, indices = torch.sort(grad_norm, dim=0, descending=True)
                            diff = sorted[:-1] - sorted[1:]
                            gumbel_noise = sample_gumbel(diff.shape, self.l2_norm_clip * 0.75 * 2 / self.args.topk_eps).to(self.args.gpu)
                            diff.add_(gumbel_noise)
                            diff.add_(self.real_parm)
                            diff_max, k = torch.topk(diff, k=1, dim=0)
 
                            grad_mask = torch.zeros(grad_norm.shape).to(self.args.gpu)
                            if diff_max > 0.75 * self.l2_norm_clip:
                                indices = indices[:k.item()+1]
                                grad_mask[indices] = 1
                                                   
                            dp_accum_grad.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(dp_accum_grad))
                            if self.args.naive == False:
                                dp_accum_grad.mul_(grad_mask)              
                            param.grad.data.add_(dp_accum_grad)
                            
                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)
            super(DPOptimizerClass, self).step(*args, **kwargs)

    return DPOptimizerClass

DPAdam = make_optimizer_class(Adam)
DPAdagrad = make_optimizer_class(Adagrad)
DPSGD = make_optimizer_class(SGD)
DPRMSprop = make_optimizer_class(RMSprop)

