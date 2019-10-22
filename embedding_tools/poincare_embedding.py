import tqdm
import torch
from torch import nn

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
        if(self.cuda):
            self.W.cuda()
        self.optimizer = optimizer_method(self.W.parameters(), lr=lr)
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
    
    def fit(self, dataloader, alpha=1.0, beta=1.0, gamma=0.0, max_iter=100,
            negative_sampling=5, pi=None, mu=None, sigma=None, distance_coef=1.):
        progress_bar = tqdm.trange(max_iter) if(self.verbose) else range(max_iter)
        for i in progress_bar:
            loss_value1, loss_value2, loss_value3 = 0,0,0
            for example, neigbhors, index_source_rw, index_context_rw in dataloader:
                self.optimizer.zero_grad()
                # obtain negatives examples sampled according to the given distribution
                with torch.no_grad():
                    negative = self.n_dist.sample(sample_shape=(index_context_rw.size(0),  negative_sampling))
                # set variables to cuda device
                if(self.cuda):
                    example = example.cuda().squeeze()
                    neigbhors = neigbhors.cuda().squeeze()
                    index_source_rw, index_context_rw = index_source_rw.cuda().squeeze(), index_context_rw.cuda().squeeze()
                    negative = negative.cuda()
                    if(pi is not None):
                        pi, mu, sigma = pi.cuda(),mu.cuda(), sigma.cuda()
                else:
                    example = example.squeeze()
                    neigbhors = neigbhors.squeeze()
                    index_source_rw, index_context_rw = index_source_rw.squeeze(), index_context_rw.squeeze()
                    negative = negative                    
                # get the needed embeddings
                embed_source, embed_neigbhor = self.W(example), self.W(neigbhors)
                embed_source_rw, embed_context_rw  = self.W(index_source_rw), self.W(index_context_rw)
                embed_negative = self.W(negative)
                # print(index_source_rw.size(),embed_source.size(), embed_neigbhor.size(), embed_source_rw.size(), embed_context_rw.size(), embed_negative.size())
                # computing O1 loss
                loss_o1 = losses.SGDLoss.O1(embed_source, embed_neigbhor, coef=distance_coef)
                # computing O2 loss
                loss_o2 = losses.SGDLoss.O2(embed_source_rw, embed_context_rw, embed_negative, coef=distance_coef)
                # computing total loss
                loss = alpha * self.agg(loss_o1) + beta * self.agg(loss_o2)
                # if we want to use the prior loss
                if(gamma > 0 and pi is not None):
                    r_example = self.W(example).squeeze()
                    pi_z = pi[example].squeeze()
                    loss_o3 = (-pi_z.detach() * torch.log(1e-4 + distribution_function.weighted_gmm_pdf(pi_z.detach(), r_example, mu.detach(), sigma.detach(), self.d)))
                    # print("loss o3 size ->", loss_o3)
                    loss += gamma * self.agg(loss_o3)
                    loss_value3 = loss_o3.sum(-1).mean().item()
                    loss_pdf3 = torch.exp(-loss_o3.mean()).item()

                loss_value1 = loss_o1.mean().item()
                loss_value2 = loss_o2.mean().item()
                loss.backward()
                self.optimizer.step()
            if(self.verbose):
                progress_bar.set_postfix({"O1":loss_value1, "O2":loss_value2, "O3":loss_value3})