'''
Code for Implicit Maximum Likelihood Estimation

This code implements the method described in the Implicit Maximum Likelihood 
Estimation paper, which can be found at https://arxiv.org/abs/1809.09087

Copyright (C) 2018    Ke Li

This code has been modified from the original version at https://people.eecs.berkeley.edu/~ke.li/projects/imle/
Modifications copyright (C) 2019-present, Royal Bank of Canada.


This file is part of the Implicit Maximum Likelihood Estimation reference 
implementation.

The Implicit Maximum Likelihood Estimation reference implementation is free 
software: you can redistribute it and/or modify it under the terms of the GNU 
Affero General Public License as published by the Free Software Foundation, 
either version 3 of the License, or (at your option) any later version.

The Implicit Maximum Likelihood Estimation reference implementation is 
distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with the Dynamic Continuous Indexing reference implementation.  If 
not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.rdp_accountant import compute_rdp, get_privacy_spent
from utils.architectures import Generator
from models.IMLE.dci_code.dci import DCI

class IMLE:
    def __init__(self, input_dim, z_dim, target_epsilon, target_delta, conditional=True):
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.generator = Generator(z_dim, input_dim, conditional).cuda()
        self.dci_db = None
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.conditional = conditional
        
    def train(self, data_np, label_np, hyperparams, private=False):
        batch_size = hyperparams.batch_size
        micro_batch_size = hyperparams.micro_batch_size
        num_batches = data_np.shape[0] // batch_size
        num_samples = num_batches * hyperparams.num_samples_factor
        sigma = hyperparams.sigma
        clip_coeff = hyperparams.clip_coeff

        class_ratios = None

        if self.conditional:
            class_ratios = torch.from_numpy(hyperparams.class_ratios)
            class_idx = dict.fromkeys([i for i in range(class_ratios.size()[0])], [])
            for i in range(label_np.shape[0]):
                class_idx[label_np[i]].append(i)

            data_np_by_class = {i : data_np[class_idx[i]] for i in range(class_ratios.size()[0])}
            rev_idx = np.concatenate([class_idx[i] for i in range(class_ratios.size()[0])], axis=0)
            self.dci_db = [DCI(data_np.shape[1], num_comp_indices = 2, num_simp_indices = 7) for _ in range(class_ratios.size()[0])]

        else:
            self.dci_db = DCI(data_np.shape[1] + 1, num_comp_indices = 2, num_simp_indices = 7)

        loss_fn = nn.MSELoss(reduction='none').cuda()
        self.generator.train()

        epsilon = 0
        steps = 0
        epoch = 0
            
        while epsilon < self.target_epsilon:
            
            if epoch % hyperparams.decay_step == 0:
                lr = hyperparams.lr * hyperparams.decay_rate ** (epoch // hyperparams.decay_step)
                optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
            
            if epoch % hyperparams.staleness == 0:
                z_np = np.empty((num_samples * batch_size, self.z_dim))
                n_cols = data_np.shape[1] if self.conditional else data_np.shape[1] + 1
                samples_np = np.empty((num_samples * batch_size, n_cols))
                categories_np = np.empty((num_samples * batch_size,1))
                for i in range(num_samples):
                    z = torch.randn(batch_size, self.z_dim).cuda()

                    if self.conditional:
                        category = torch.multinomial(class_ratios,  batch_size, replacement=True).unsqueeze(1).cuda().float()
                        categories_np[i * batch_size:(i + 1) * batch_size] = category.cpu().data.numpy()
                        samples = self.generator(torch.cat([z, category], dim=1))
                    else:
                        samples = self.generator(z)

                    z_np[i*batch_size:(i+1)*batch_size] = z.cpu().data.numpy()
                    samples_np[i*batch_size:(i+1)*batch_size] = samples.cpu().data.numpy()


                nearest_indices = []

                if self.conditional:
                    for i in range(len(self.dci_db)):
                        self.dci_db[i].reset()
                        self.dci_db[i].add(samples_np[[j for j in range(categories_np.shape[0]) if categories_np[j] == i]],
                                           num_levels=2, field_of_view=10, prop_to_retrieve=0.002)

                        cur_nearest_indices, _ = self.dci_db[i].query(data_np_by_class[i], num_neighbours = 1, field_of_view = 20, prop_to_retrieve = 0.02)
                        nearest_indices.append(np.array(cur_nearest_indices)[:, 0])

                    z_np = z_np[np.concatenate(nearest_indices, axis=0)[rev_idx]]
                else:
                    self.dci_db.reset()
                    self.dci_db.add(samples_np, num_levels=2, field_of_view=10, prop_to_retrieve=0.002 )
                    cur_nearest_indices, _ = self.dci_db.query(np.concatenate([data_np, np.expand_dims(label_np, axis=1)], axis=1)
                                                               , num_neighbours = 1, field_of_view = 20, prop_to_retrieve = 0.02)
                    nearest_indices.append(np.array(cur_nearest_indices)[:, 0])

                    z_np = z_np[np.concatenate(nearest_indices, axis=0)]

                z_np += 0.01*np.random.randn(*z_np.shape)

                del samples_np
            
            err = 0.
            for i in range(num_batches):
                self.generator.zero_grad()
                cur_z = torch.from_numpy(z_np[i*batch_size:(i+1)*batch_size]).float().cuda()
                cur_data = torch.from_numpy(data_np[i*batch_size:(i+1)*batch_size]).float().cuda()
                cur_labels = torch.from_numpy(label_np[i*batch_size:(i+1)*batch_size]).float().cuda()

                if self.conditional:
                    cur_samples = self.generator(torch.cat([cur_z, cur_labels.unsqueeze(1)], dim=1))
                    loss = loss_fn(cur_samples, cur_data).mean(1)
                else:
                    cur_samples = self.generator(cur_z)
                    loss = loss_fn(cur_samples, torch.cat([cur_data, cur_labels.unsqueeze(1)], dim=1)).mean(1)


                if private:
                    clipped_grads = {name: torch.zeros_like(param) for name, param in self.generator.named_parameters()}
                    for k in range(int(loss.size(0)/micro_batch_size)):
                        loss_micro = loss[k*micro_batch_size : (k+1)*micro_batch_size].mean(0).view(1)
                        loss_micro.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), clip_coeff)
                        for name, param in self.generator.named_parameters():
                            clipped_grads[name] += param.grad
                        self.generator.zero_grad()

                    for name, param in self.generator.named_parameters():

                        # add noise here
                        param.grad = (clipped_grads[name] + torch.Tensor(
                            clipped_grads[name].size()).normal_(0, sigma*clip_coeff).cuda()) / (loss.size(0)/micro_batch_size)

                    steps += 1
                else:
                    loss.mean(0).view(1).backward()

                err += loss.mean(0).item()
                optimizer.step()

            epoch += 1
            if private:
                max_lmbd = 4095
                lmbds = range(2, max_lmbd + 1)
                rdp = compute_rdp(batch_size / data_np.shape[0], sigma, steps, lmbds)
                epsilon, _, _ = get_privacy_spent(lmbds, rdp, target_delta=1e-5)
            else:
                if epoch > hyperparams.num_epochs:
                    epsilon = np.inf

            print("Epoch %d: Error: %f Epsilon spent: %f" % (epoch, err / num_batches, epsilon))

    def generate(self, num_rows, class_ratios, batch_size=1000):
        steps = num_rows // batch_size
        synthetic_data = []
        if self.conditional:
            class_ratios = torch.from_numpy(class_ratios)
        for step in range(steps):
            noise = torch.randn(batch_size, self.z_dim).cuda()
            if self.conditional:
                cat = torch.multinomial(class_ratios,  batch_size, replacement=True).unsqueeze(1).cuda().float()
                synthetic = self.generator(torch.cat([noise, cat], dim=1))
                synthetic = torch.cat([synthetic, cat], dim=1)

            else:
                synthetic = self.generator(noise)

            synthetic_data.append(synthetic.cpu().data.numpy())

        if steps*batch_size < num_rows:
            noise = torch.randn(num_rows - steps*batch_size, self.z_dim).cuda()

            if self.conditional:
                cat = torch.multinomial(class_ratios, num_rows - steps*batch_size, replacement=True).unsqueeze(1).cuda().float()
                synthetic = self.generator(torch.cat([noise, cat], dim=1))
                synthetic = torch.cat([synthetic, cat], dim=1)
            else:
                synthetic = self.generator(noise)
            synthetic_data.append(synthetic.cpu().data.numpy())

        return np.concatenate(synthetic_data)
