import torch 
import math
import numpy as np
import tqdm
def o_1_loss(me, mw, distance):
    return  -(torch.log(torch.exp(-distance(me, mw)))).sum(-1).sum(-1).mean()

def learn_init(m, dataloader, dataset,  optimizer, distance, max_iter=20, alpha=1, beta=1, cuda=False, verbose=True):
    progress_bar = tqdm.trange(max_iter) if(verbose) else range(max_iter)
    for i in progress_bar:
        loss_value1 = 0
        loss_value2 = 0
        
        for example, neigbhors, walks in dataloader:
            optimizer.zero_grad()
            # Computing l
            # oss function
            # O_1
            if(cuda):
                example = example.cuda()
                neigbhors = neigbhors.cuda()
                walks = walks.cuda()
            r_example = example.unsqueeze(1).expand_as(neigbhors)
            me, mw = m(r_example), m(neigbhors)

            loss_o1 = -(torch.log(torch.exp(-distance(me, mw)))).sum(-1).sum(-1).mean()


            # O_2        
            r_example = example.unsqueeze(1).expand_as(walks)
            me, mw = m(r_example), m(walks)
            positive_d = (distance(me, mw))
            me = me.expand(walks.size(0), walks.size(1),  5, mw.size(-1))
            negative = (torch.rand(walks.size(0), walks.size(1), 5) * len(dataset))
            if(cuda):
                negative = negative.cuda()
            negative = m(negative.long())

            negative_d = distance(me, negative)

            loss_o2 = torch.log( 1 + (torch.exp(-(negative_d - positive_d.expand_as(negative_d)))).sum(-1)).mean()

            loss = alpha * loss_o1 + beta * loss_o2 

            loss_value1 = loss_o1.item()
            loss_value2 = loss_o2.item()
            loss.backward()
            optimizer.step()
        if(verbose):
            progress_bar.set_postfix({"loss":beta *loss_value2})
        # print("O_1 : ", loss_value1)
        # print("O_2 : ", loss_value2)

pi_2_3 = pow((2*math.pi),2/3)
a_for_erf = 8.0/(3.0*np.pi)*(np.pi-3.0)/(4.0-np.pi)

def erf_approx(x):
    return torch.sign(x)*torch.sqrt(1-torch.exp(-x*x*(4/np.pi+a_for_erf*x*x)/(1+a_for_erf*x*x)))



def weighted_gmm_pdf(w, z, mu, sigma, distance):    
    eps = 1e-4
    z_u = z.unsqueeze(1).expand(z.size(0), len(mu), z.size(1))
    mu_u = mu.unsqueeze(0).expand_as(z_u)
    distance_to_mean = distance(z_u, mu_u)
    sigma_u = sigma.unsqueeze(0).expand_as(distance_to_mean)
    distribution_normal = torch.exp(-((distance_to_mean)**2)/(2 * sigma_u**2))

    norm_fact = 1.
    # print(distribution_normal.mean())
    # print(distribution_normal.min())
    return w * ((distribution_normal+eps)/ norm_fact)

def learn(m, dataloader, dataset,  optimizer, distance, pi, mu, sigma, max_iter=20, alpha=1, beta=1, gamma=1,  cuda=False, verbose=True):
    progress_bar = tqdm.trange(max_iter) if(verbose) else range(max_iter)
    for i in progress_bar:
        loss_value1 = 0
        loss_value2 = 0
        loss_value3 = 0
        for example, neigbhors, walks in dataloader:
            optimizer.zero_grad()
            # Computing loss function
            # O_1
            if(m.weight.mean() != m.weight.mean()):
                a = i
                mu_norm = mu.norm(2,-1)
                print("Iteration ", i)
                
                assert(m.weight.mean() == m.weight.mean())


            if(cuda):
                example = example.cuda()
                neigbhors = neigbhors.cuda()
                walks = walks.cuda()
                pi = pi.cuda()
                sigma = sigma.cuda()
                mu = mu.cuda()
            r_example = example.unsqueeze(1).expand_as(neigbhors)
            me, mw = m(r_example), m(neigbhors)
            # print("exemple size : ",me.size())
            # print("exemple size : ",mw.size())
            loss_o1 = -(torch.log(torch.exp(-distance(me, mw)))).sum(-1).sum(-1).mean()
            # print(loss_o1.size())

            # O_2        
            r_example = example.unsqueeze(1).expand_as(walks)
            me, mw = m(r_example), m(walks)
            positive_d = (distance(me, mw))
            # print(me.size())
            me = me.expand(walks.size(0), walks.size(1),  5, mw.size(-1))
            negative = (torch.rand(walks.size(0), walks.size(1), 5) * len(dataset))
            if(cuda):
                negative = negative.cuda()
            negative = m(negative.long())
            # print("qsdf",me.size())
            negative_d = distance(me, negative)

            loss_o2 = torch.log( 1 + (torch.exp(-(negative_d - positive_d.expand_as(negative_d)))).sum(-1)).mean()
            r_example = m(example).squeeze()
            p_example = pi.unsqueeze(0).expand(len(example), len(mu))
            loss_o3 = (-torch.log( weighted_gmm_pdf(p_example, r_example, mu, sigma, distance))).mean()
            # print(loss_o1, loss_o2, loss_o3)
            loss = alpha * loss_o1 + beta * loss_o2 + gamma * loss_o3
            
            loss_value1 = loss_o1.item()
            loss_value2 = loss_o2.item()
            loss_value3 = loss_o3.item()
            if(loss_value3 != loss_value3 or loss_value3 == math.inf):
                print("An error occured")
                print("O_1 : ", loss_value1)
                print("O_2 : ", loss_value2)            
                print("O_3 : ", loss_value3)  
            loss.backward()
            optimizer.step()
        # print("O_1 : ", loss_value1)
        # print("O_2 : ", loss_value2)            
        # print("O_3 : ", loss_value3)  
        if(verbose):
            progress_bar.set_postfix({"loss":alpha *loss_value1+beta * loss_value2 + gamma * loss_value3})