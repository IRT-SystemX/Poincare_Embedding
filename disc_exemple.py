import torch
from torch import nn
from torch.utils.data import DataLoader

from data_tools import corpora_tools
from data_tools import corpora
from data_tools import data_tools
from embedding_tools import hyperbolic_embeddings_graph
from function_tools import modules
from optim_tools import optimizer
from visualisation_tools import  plot_tools
from evaluation_tools import  evaluation
from em_tools import em_original
from em_tools import em_euclidean
from function_tools.numpy_function import RiemannianFunction


parameters = type('', (), {})()
parameters.dataset_loading_func = corpora.load_football
parameters.lr = 2
parameters.max_em_iter = 20
parameters.max_proj_iter = 20
parameters.walk_length = 4
parameters.n_gaussian = 12
parameters.cuda = False
print("Loading Corpus ")
D, X, Y = parameters.dataset_loading_func()

print("Create embeddings table")
embeddings = nn.Embedding(len(D), 2, max_norm=1.0-1e-4)
embeddings.weight.data[:] = embeddings.weight.data[:]*1e-1
if(parameters.cuda):
    embeddings.cuda()
optimizer_method = optimizer.PoincareBallSGD(embeddings.parameters(), lr=parameters.lr)

print("Creating dataset")
# index of examples dataset
dataset_index = corpora_tools.from_indexable(torch.arange(0,len(D),1).unsqueeze(-1))
D.set_path(False)
# random walk dataset
d_rw = D.light_copy()
d_rw.set_walk(parameters.walk_length, 1.0)
d_rw.set_path(True)
# neigbhor dataset
d_v = D.light_copy()
d_v.set_walk(1, 1.0)

print("Merging dataset")
embedding_dataset = corpora_tools.zip_datasets(dataset_index,
                                                corpora_tools.select_from_index(d_rw, element_index=0),
                                                corpora_tools.select_from_index(d_v, element_index=0)
                                                )
training_dataloader = DataLoader(embedding_dataset, 
                            batch_size=128, 
                            shuffle=True,
                            num_workers=4,
                            collate_fn=data_tools.PadCollate(dim=0),
                            drop_last=False
                    )

print("Initialize embeddings")
hyperbolic_embeddings_graph.learn_init(embeddings, training_dataloader, 
                                embedding_dataset,  optimizer_method, 
                                modules.hyperbolicDistance, 
                                max_iter=150, alpha=.1, beta=10., cuda=parameters.cuda
                            )
print("First EM Pass")
em_object = em_original.RiemannianEM(parameters.n_gaussian, RiemannianFunction.riemannian_distance, init_mod="kmeans")
em_object.fit(embeddings.weight.data.cpu(), max_iter=5)
pi_k, mu, sigma = em_object.getParameters()

print("pi_k : ", pi_k)
print("mu : ", mu)
print("sigma : ", sigma)

for i in range(2):
        hyperbolic_embeddings_graph.learn(embeddings, training_dataloader, 
                                embedding_dataset,  optimizer_method, 
                                modules.hyperbolicDistance, pi_k.detach(), mu.detach(), sigma.detach(),
                                max_iter=50, alpha=.01, beta=.1, gamma=.0,  cuda=parameters.cuda
        )
        em_object.fit(embeddings.weight.data.cpu(),max_iter=5)
        pi_k, mu, sigma = em_object.getParameters()

        print("pi_k : ", pi_k)
        print("mu : ", mu)
        print("sigma : ", sigma)

print("End of learning : ")
print("pi_k : ", pi_k)
print("mu : ", mu)
print("sigma : ", sigma)
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
plt.figure() 

embeddings.cpu()

import numpy as np
unique_label = np.unique(sum([ y for k, y in D.Y.items()],[]))
print(unique_label)
for i in range(len(embeddings.weight)):
    color = plt_colors.hsv_to_rgb([D.Y[i][0]/(len(unique_label)),0.5,0.8])
    #print(D.Y[i])
    plt.scatter(embeddings.weight.data[i,0].numpy(), embeddings.weight.data[i,1].numpy(), c = [color])
plt.scatter(mu[0][0], mu[0][1], c = "r")
plt.scatter(mu[1][0], mu[1][1], c = "r")
plt.savefig("figures/embeddings_node.pdf", format="pdf")
print("saved")
colors = []
for i in range(len(embeddings.weight)):
    colors.append(plt_colors.hsv_to_rgb([D.Y[i][0]/(len(unique_label)),0.5,0.8]))

plot_tools.plot_embedding_distribution(embeddings.weight.data, pi_k, mu, sigma, colors=colors)

print("Evaluation : ")
print("Performances -> " , evaluation.accuracy_cross_validation(embeddings.weight.data, D.Y, pi_k,  mu, sigma, 5))
