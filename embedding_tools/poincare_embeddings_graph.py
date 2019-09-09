import tqdm
import torch
from torch import nn

from function_tools import poincare_function, poincare_module, gmm_tools
from optim_tools import optimizer



class RiemannianEmbedding(nn.Module):
    def __init__(self, n_exemple, cuda=False, lr=1e-2, verbose=True, negative_distribution=None,
                optimizer_method=optimizer.PoincareBallSGDAdd):
        super(RiemannianEmbedding, self).__init__()
        self.cuda = cuda
        self.N = n_exemple
        self.W = poincare_module.PoincareEmbedding(n_exemple, 2)
        if(self.cuda):
            self.W.cuda()
        self.optimizer = optimizer_method(self.W.parameters(), lr=lr)
        self.verbose = verbose
        self.d = poincare_function.poincare_distance
        if(negative_distribution is None):
            self.n_dist = torch.distributions.Categorical(torch.ones(self.N)/self.N)
        else:
            self.n_dist = negative_distribution

    def forward(self, x):
        return self.W(x)

    def get_PoincareEmbeddings(self):
        return self.W.l_embed(torch.arange(0, self.N, device=self.W.l_embed.weight.data.device)).detach()

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr
    
    def fit(self, dataloader, alpha=1.0, beta=1.0, gamma=0.0, pi=None, mu=None, sigma=None, max_iter=100,
            negative_sampling=5):
        
        if(pi is None):
            gamma = 0.0

        progress_bar = tqdm.trange(max_iter) if(self.verbose) else range(max_iter)
        for i in progress_bar:
            loss_value1, loss_value2, loss_value3, loss_pdf3 = 0,0,0,0
            for example, neigbhors, walks in dataloader:
                # print(example.size(), neigbhors.size(), walks.size())
                self.optimizer.zero_grad()
                if(self.cuda):
                    example = example.cuda()
                    neigbhors = neigbhors.cuda()
                    walks = walks.cuda()
                    if(pi is not None):
                        pi = pi.cuda()
                        sigma = sigma.cuda()
                        mu = mu.cuda()
                r_example = example.unsqueeze(1).expand_as(neigbhors)
                me, mw = self.W(r_example), self.W(neigbhors)

                loss_o1 = -(torch.log(torch.exp(-self.d(me, mw)))).mean()

                mw = self.W(walks)
                positive_d = (self.d(mw[:,:,0], mw[:,:,1]))

                with torch.no_grad():
                    negative = self.n_dist.sample(sample_shape=(walks.size(0), walks.size(1), negative_sampling))
                if(self.cuda):
                    negative = negative.cuda()
                negative = self.W(negative.long())
                negative_d = self.d(mw[:,:,0].unsqueeze(2).expand_as(negative), negative)
                # print(torch.log( 1 + (torch.exp(-(negative_d - positive_d.unsqueeze(-1).expand_as(negative_d)))).sum(-1)).size())
                loss_o2 = torch.log( 1 + (torch.exp(-(negative_d - positive_d.unsqueeze(-1).expand_as(negative_d)))).sum(-1)).mean()
                loss = alpha * loss_o1 + beta * loss_o2 
                if(gamma > 0):
                    r_example = self.W(example).squeeze()
                    pi_z = pi[example].squeeze()
                    loss_o3 = (-pi_z.detach() * torch.log(1e-4 + gmm_tools.weighted_gmm_pdf(pi_z.detach(), r_example, mu.detach(), sigma.detach(), self.d)))
                    # print("loss o3 size ->", loss_o3)
                    loss += gamma * loss_o3.mean()
                    loss_value3 = loss_o3.sum(-1).mean().item()
                    loss_pdf3 = torch.exp(-loss_o3.mean()).item()

                loss_value1 = loss_o1.item()
                loss_value2 = loss_o2.item()
                loss.backward()
                self.optimizer.step()
            if(self.verbose):
                progress_bar.set_postfix({"O1":alpha*loss_value1, "O2":beta *loss_value2, "O3":gamma *loss_value3, "PDF":loss_pdf3})