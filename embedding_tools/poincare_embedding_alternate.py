import tqdm
import torch
from torch import nn
import random

from function_tools import poincare_function, poincare_module, distribution_function
from embedding_tools import losses
from optim_tools import optimizer



class PoincareEmbedding(nn.Module):
    def __init__(self, n_exemple, size=10,  cuda=False, lr=1e-2, verbose=True, negative_distribution=None,
                optimizer_method=optimizer.PoincareBallSGDAdd, aggregation=torch.sum):
        super(PoincareEmbedding, self).__init__()
        self.cuda = cuda
        self.N = n_exemple
        self.W = poincare_module.PoincareEmbedding(n_exemple, size)
        self.C = poincare_module.PoincareEmbedding(n_exemple, size)
        if(self.cuda):
            self.W.cuda()
            self.C.cuda()
        self.optimizer = optimizer_method(list(self.W.parameters()) + list(self.C.parameters()), lr=lr)
        self.verbose = verbose
        self.d = poincare_function.poincare_distance
        if(negative_distribution is None):
            self.n_dist = torch.distributions.Categorical(torch.ones(self.N)/self.N)
        else:
            self.n_dist = negative_distribution
        self.agg = aggregation
    

    def forward(self, x):
        return self.W(x)

    def get_PoincareEmbeddings(self):
        return self.W(torch.arange(0, self.N, device=self.W.l_embed.weight.data.device)).detach()

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr
    
    def fit(self, dataloader_o1, dataloader_o2, dataloader_o3, alpha=1.0, beta=1.0, gamma=0.0, max_iter=100,
            negative_sampling=5, pi=None, mu=None, sigma=None, normalisation_coef=None,  distance_coef=1.
            , log_callback=None, logger=None):

        self.log_callback = log_callback
        self.logger = logger   

        lcres = {}
        calback_log_list = []

        progress_bar = tqdm.trange(max_iter) if(self.verbose) else range(max_iter)
        if(self.cuda and gamma > 0 and pi is not None):
            pi = pi.cuda()
            mu = mu.cuda()
            sigma = sigma.cuda()
            normalisation_coef = normalisation_coef.cuda()
        negative_all = None


        for i in progress_bar:
            loss_value1, loss_value2, loss_value3 = 0,0,0

            # reset all progress_bar
            pb_O1 = tqdm.trange(len(dataloader_o1))
            pb_O2 = tqdm.trange(len(dataloader_o2))
            pb_O3 = tqdm.trange(len(dataloader_o3))
            #SGD on O_2
            o_lval = 0 
            for pb_i, (example_index_a, example_index_b) in zip(pb_O2, dataloader_o2):
                self.optimizer.zero_grad()
                # getting negative examples
                if(negative_all is None):
                    negative_all = self.n_dist.sample( sample_shape=(1000,2580,  negative_sampling))
                    if(self.cuda):
                        negative_all = negative_all.cuda()
                if(self.cuda):
                    example_index_a = example_index_a.cuda()
                    example_index_b = example_index_b.cuda()
                negative = negative_all[random.randint(0, 999)][:example_index_a.size(0)]

                #getting embedding
                # print(example_index_a.size(), example_index_b.size(), negative.size())
                example_embedding_a, example_embedding_b = (self.W(example_index_a).squeeze(), 
                                                            self.C(example_index_b).squeeze())
                negative_embedding = self.C(negative).detach()
                loss_o2 = losses.SGDLoss.O2(example_embedding_a, example_embedding_b,
                                            negative_embedding, coef=distance_coef)
                loss_value2 += loss_o2.detach().sum().item()
                o_lval += loss_o2.detach().mean().item()
                if(pb_i%200 == 0):
                    pb_O2.set_postfix({"LO2":o_lval/200 })
                    o_lval = 0
                self.agg(beta * loss_o2).backward()
                self.optimizer.step()

            # SGD on O_1
            o_lval = 0 
            for pb_i, (example_index_a, example_index_b) in zip(pb_O1, dataloader_o1):
                if(self.cuda):
                    example_index_a = example_index_a.cuda()
                    example_index_b = example_index_b.cuda()
                self.optimizer.zero_grad()
                example_embedding_a, example_embedding_b = self.W(example_index_a), self.W(example_index_b)
                loss_o1 = losses.SGDLoss.O1(example_embedding_a, example_embedding_b, coef=distance_coef)
                loss_value1 += loss_o1.detach().sum().item()
                self.agg(alpha * loss_o1).backward()
                o_lval += loss_o1.detach().mean().item()
                if(pb_i%100 == 0):
                    pb_O1.set_postfix({"LO1":o_lval/100 })
                    o_lval = 0
                self.optimizer.step()





            # sgd on O_3
            if(gamma > 0 and pi is not None):
                for example_index, *others in dataloader_o3:
                    self.optimizer.zero_grad()
                    if(self.cuda):
                        example_index = example_index.cuda()
                    # getting embeddings (BxD)
                    example_embedding = self.W(example_index).squeeze()
                    # getting gmm weigth (BxM)
                    gmm_weight = pi[example_index]
                    loss_o3 = losses.SGDLoss.O3(example_embedding, gmm_weight, mu, sigma, normalisation_coef)
                    # print(loss_o3.sum(-1))
                    loss_value3 += loss_o3.detach().sum().item()
                    (gamma *self.agg(loss_o3)).backward()
                    self.optimizer.step()
            if(self.log_callback is not None and i%50==0):
                lcres = self.log_callback()
                if(self.logger is not None):
                    calback_log_list.append(lcres)
                    self.logger["calback-log"] = calback_log_list
            if(self.verbose):
                loss_value1 = loss_value1/len(dataloader_o1.dataset)
                loss_value2 = loss_value2/len(dataloader_o2.dataset)
                loss_value3 = loss_value3/len(dataloader_o3.dataset)
                progress_bar.set_postfix(dict({"O1":loss_value1, "O2":loss_value2, "O3":loss_value3}).items() | lcres.items())
