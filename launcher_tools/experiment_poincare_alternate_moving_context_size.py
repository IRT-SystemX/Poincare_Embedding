import argparse
import tqdm

import torch
from function_tools import pytorch_categorical
from torch.utils.data import DataLoader
import os
from multiprocessing import Process, Manager
from embedding_tools.poincare_embedding_alternate import PoincareEmbedding as PEmbed
from clustering_tools.poincare_em import PoincareEM as PEM

from callback_tools.callback import log_callback_em_conductance, log_callback_kmeans_conductance
from callback_tools import tools as callback_tools
from data_tools import corpora_tools
from data_tools import corpora
from data_tools import data
from evaluation_tools import evaluation
from visualisation_tools import plot_tools

from optim_tools import optimizer
import random 
import numpy as np
from data_tools import config, logger


parser = argparse.ArgumentParser(description='Start an experiment')

parser.add_argument('--init-lr', dest="init_lr", type=float, default=-1.0,
                    help="Learning rate for the first embedding step")
parser.add_argument('--lr', dest="lr", type=float, default=5e-1,
                    help="learning rate for embedding")
parser.add_argument('--init-alpha', dest="init_alpha", type=float, default=-1.0,
                    help="alpha for the first embedding step")
parser.add_argument('--alpha', dest="alpha", type=float, default=1e-2,
                    help="alpha for embedding")
parser.add_argument('--init-beta', dest="init_beta", type=float, default=-1.0,
                    help="beta for the first embedding step")
parser.add_argument('--beta', dest="beta", type=float, default=1,
                    help="beta for embedding")
parser.add_argument('--gamma', dest="gamma", type=float, default=1e-1,
                    help="gamma rate for embedding")
parser.add_argument('--n-gaussian', dest="n_gaussian", type=int, default=2,
                    help="number of gaussian for EM algorithm")
parser.add_argument('--dataset', dest="dataset", type=str, default="karate",
                    help="dataset to use for the experiments")
parser.add_argument('--walk-lenght', dest="walk_lenght", type=int, default=20,
                    help="size of random walk")
parser.add_argument('--cuda', dest="cuda", action="store_true", default=False,
                    help="using GPU for operation")
parser.add_argument('--epoch', dest="epoch", type=int, default=2,
                    help="number of loops alternating embedding/EM")
parser.add_argument('--epoch-embedding-init', dest="epoch_embedding_init", type=int, default=100,
                    help="maximum number of epoch for first embedding gradient descent")
parser.add_argument('--epoch-embedding', dest="epoch_embedding", type=int, default=10,
                    help="maximum number of epoch for embedding gradient descent")
parser.add_argument('--id', dest="id", type=str, default="0",
                    help="identifier of the experiment")
parser.add_argument('--save', dest="save", action="store_true", default=True,
                    help="saving results and parameters")
parser.add_argument('--precompute-rw', dest='precompute_rw', type=int, default=-1,
                    help="number of random path to precompute (for faster embedding learning) if negative \
                        the random walks is computed on flight")
parser.add_argument('--context-size', dest="context_size", type=int, default=5,
                    help="size of the context used on the random walk")
parser.add_argument("--negative-sampling", dest="negative_sampling", type=int, default=10,
                    help="number of negative samples for loss O2")
parser.add_argument("--embedding-optimizer", dest="embedding_optimizer", type=str, default="exphsgd", 
                    help="the type of optimizer used for learning poincaré embedding")
parser.add_argument("--em-iter", dest="em_iter", type=int, default=10,
                    help="Number of EM iterations")
parser.add_argument("--size", dest="size", type=int, default=3,
                    help="dimenssion of the ball")
parser.add_argument("--batch-size", dest="batch_size", type=int, default=10000,
                    help="Batch number of elements")
parser.add_argument("--seed", dest="seed", type=int, default=42,
                    help="the seed used for sampling random numbers in the experiment")  
parser.add_argument('--force-rw', dest="force_rw", action="store_false", default=True,
                    help="if set will automatically compute a new random walk for the experiment") 
parser.add_argument('--loss-aggregation', dest="loss_aggregation", type=str, default="sum",
                    help="The type of loss aggregation sum or mean")            
parser.add_argument('--distance-coef', dest="distance_coef", type=float, default=1.,
                    help="Factor applied to the distance in the loss")
parser.add_argument('--reset-em', dest="reset_em", action="store_true", default=False,
                    help="reset the em parameters at each iteration")         
parser.add_argument('--dataset-type', dest="dataset_type", type=str, default="FlatCorpus",
                    help="type of dataset")                          
args = parser.parse_args()

# set the seed for random sampling
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(a=args.seed)

global_config = config.ConfigurationFile("./data/config.conf")
saving_folder = global_config["save_folder"]


dataset_dict = { "karate": corpora.load_karate,
            "football": corpora.load_football,
            "flickr": corpora.load_flickr,
            "dblp": corpora.load_dblp,
            "books": corpora.load_books,
            "blogCatalog": corpora.load_blogCatalog,
            "wikipedia": corpora.load_wikipedia,
            "polblog": corpora.load_polblogs,
            "adjnoun": corpora.load_adjnoun
          }
dataset_type = { "WeightedFlatCorpus": corpora.load_karate,
            "FlatCorpus": corpora.load_football
          }
aggregation_dict = { "sum": torch.sum, "mean": torch.mean}

optimizer_dict = {"addhsgd": optimizer.PoincareBallSGDAdd,
                    "exphsgd": optimizer.PoincareBallSGDExp,
                    "hsgd": optimizer.PoincareBallSGD}


print("The following options are use for the current experiment ", args)
# check if dataset exists

if(args.dataset not in dataset_dict):
    print("Dataset " + args.dataset + " does not exist, please select one of the following : ")
    print(list(dataset_dict.keys()))
    quit()

if(args.loss_aggregation not in aggregation_dict):
    print("Aggregation method " + args.dataset + " does not exist, please select one of the following : ")
    print(list(aggregation_dict.keys()))
    quit()
if(args.embedding_optimizer not in optimizer_dict):
    print("Optimizer " + args.embedding_optimizer + " does not exist, please select one of the following : ")
    print(list(optimizer_dict.keys()))
    quit()

if(args.init_lr <= 0):
    args.init_lr = args.lr
if(args.init_alpha < 0):
    args.init_alpha = args.alpha
if(args.init_beta < 0):
    args.init_beta = args.beta

alpha, beta = args.init_alpha, args.init_beta

print("Loading Corpus ")
D, X, Y = dataset_dict[args.dataset]()
print("Creating dataset")
# index of examples dataset
dataset_index = corpora_tools.from_indexable(torch.arange(0,len(D),1).unsqueeze(-1))
print("Dataset Size : ", len(D))
print("log will be saved at : ",os.path.join(saving_folder,args.id+"/"))
if(args.save):
    os.makedirs(os.path.join(saving_folder,args.id+"/"), exist_ok=True)
    logger_object = logger.JSONLogger(os.path.join(saving_folder,args.id+"/log.json"))
    logger_object.append(vars(args))


D.set_path(False)

# negative sampling distribution
frequency = D.getFrequency()**(3/4)
frequency[:,1] /= frequency[:,1].sum()
frequency =  pytorch_categorical.Categorical(frequency[:,1])
# random walk dataset
d_rw = D.light_copy()


print("Loading Neighbor corpus")
dataset_o1 = corpora.NeigbhorFlatCorpus(X, Y)
print("Loading Context corpus")
dataset_o2 = corpora.RandomContextSizeFlat(X, Y, precompute=args.precompute_rw, 
                path_len=args.walk_lenght, context_size=args.context_size)
print("Creating Dataset index")
dataset_o3 = dataset_index
# print(d_rw[1][0].size())
# print(len(embedding_dataset[0]))
# print(embedding_dataset[29][-1][20:25])

# if(args.cuda):
#     dataset_o1.cuda()
#     d_rw.cuda()
def collate_fn_simple(my_list):
    v =  torch.cat(my_list,0)
    return v[:,0], v[:,1]

training_dataloader_o1 = data.RawDataloader(dataset_o1, 
                            batch_size=args.batch_size, 
                            shuffle=True
                    )
print("icic",len(training_dataloader_o1))

training_dataloader_o2 = DataLoader(dataset_o2, 
                            batch_size=args.batch_size, 
                            shuffle=True,
                            num_workers=10,
                            collate_fn=collate_fn_simple
                    )
training_dataloader_o3 = DataLoader(dataset_o3, 
                            batch_size=args.batch_size, 
                            shuffle=True,
                            num_workers=10,
                            drop_last=False
                    )

print("Dataset O1 size (number edges)   : ", len(dataset_o1))
print("Dataset O2 size (number context) : ", len(dataset_o2))
print("Dataset O3 size (number nodes)   : ", len(dataset_o3))

representation_d = []
pi_d = []
mu_d = []
sigma_d = []
# if dimension is 2 we can plot 
# we store colors here
if(args.size == 2):
    import matplotlib.pyplot as plt
    import matplotlib.colors as plt_colors
    import numpy as np
    unique_label = np.unique(sum([ y for k, y in D.Y.items()],[]))
    colors = []

    for i in range(len(D.Y)):
        colors.append(plt_colors.hsv_to_rgb([D.Y[i][0]/(len(unique_label)),0.5,0.8]))


alpha, beta = args.init_alpha, args.init_beta
embedding_alg = PEmbed(len(dataset_index), size=args.size, lr=args.init_lr, cuda=args.cuda, negative_distribution=frequency,
                        optimizer_method=optimizer_dict[args.embedding_optimizer], aggregation=aggregation_dict[args.loss_aggregation])
em_alg = PEM( args.n_gaussian, init_mod="kmeans-hyperbolic", verbose=False)

# create a callback function to log the conductence
callback_function = callback_tools.generic_callback({"embeddings": embedding_alg.get_PoincareEmbeddings}, 
                                {"adjancy_matrix":X, "n_centroid": args.n_gaussian},
                                log_callback_kmeans_conductance)

pi, mu, sigma, normalisation_factor = None, None, None, None
pik = None
epoch_embedding = args.epoch_embedding_init
pb = tqdm.trange(args.epoch)

for i in pb:

    embedding_alg.fit(training_dataloader_o1, training_dataloader_o2, training_dataloader_o3,
                        alpha=alpha, beta=beta, gamma=args.gamma, max_iter=epoch_embedding,
                        pi=pik, mu=mu, sigma=sigma, negative_sampling=args.negative_sampling,
                        distance_coef=args.distance_coef, normalisation_coef=normalisation_factor,
                        log_callback=callback_function, logger=logger_object)

    if(i==0):
        embedding_alg.set_lr(args.lr)
        alpha, beta = args.alpha, args.beta
        epoch_embedding = args.epoch_embedding
        torch.save(embedding_alg.get_PoincareEmbeddings().cpu(), os.path.join(saving_folder,args.id+"/embeddings_init.t7"))
        
    if(args.reset_em):
        em_alg = PEM(args.n_gaussian, init_mod="kmeans-hyperbolic", verbose=True)
    em_alg.fit(embedding_alg.get_PoincareEmbeddings().cpu(), max_iter=args.em_iter)
    pi, mu, sigma = em_alg.get_parameters()
    pik = em_alg.get_pik(embedding_alg.get_PoincareEmbeddings().cpu())
    normalisation_factor = em_alg.get_normalisation_coef()
    total_accuracy = evaluation.poincare_unsupervised_em(embedding_alg.get_PoincareEmbeddings().cpu(), D.Y, args.n_gaussian, em=em_alg, verbose=False)
    logger_object.append({"accuracy_iter_"+str(i): total_accuracy})
    pb.set_postfix({"perfomance": total_accuracy})
    if(args.size == 2):
        plot_tools.plot_embedding_distribution_multi([embedding_alg.get_PoincareEmbeddings().cpu()], 
                                                        [pi], [mu],  [sigma], 
                                                    labels=None, N=100, colors=colors, 
                                                    save_path=os.path.join(saving_folder, args.id+"/fig_epoch_"+str(i)+".png"))
representation_d.append(embedding_alg.get_PoincareEmbeddings().cpu())
pi_d.append(pi)
mu_d.append(mu)
sigma_d.append(sigma)



#evaluate performances on all disc
total_accuracy = evaluation.poincare_unsupervised_em(embedding_alg.get_PoincareEmbeddings().cpu(), D.Y,
                                                     args.n_gaussian, em=em_alg, verbose=False)

print("\nPerformances joined -> " ,
    total_accuracy
)
logger_object.append({"accuracy": total_accuracy})

if(args.save):
    import matplotlib.pyplot as plt
    import matplotlib.colors as plt_colors
    import numpy as np
    torch.save(representation_d, os.path.join(saving_folder,args.id+"/embeddings.t7"))
    torch.save( {"pi": pi_d, "mu":mu_d, "sigma":sigma_d},os.path.join(saving_folder,args.id+"/pi_mu_sigma.t7"))


    if(args.size == 2):

        plot_tools.plot_embedding_distribution_multi(representation_d, pi_d, mu_d,  sigma_d, 
                                                    labels=None, N=100, colors=colors, 
                                                    save_path=os.path.join(saving_folder,args.id+"/fig.png"))


    print({"pi": pi_d, "mu":mu_d, "sigma":sigma_d})
