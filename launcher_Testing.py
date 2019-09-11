import argparse
import tqdm

import torch
import pytorch_categorical
from torch.utils.data import DataLoader
import os
from multiprocessing import Process, Manager
from embedding_tools.poincare_embeddings_graph import RiemannianEmbedding as PEmbed
from em_tools.em_original import RiemannianEM as PEM
from data_tools import corpora_tools
from data_tools import corpora
from data_tools import data_tools
from evaluation_tools import evaluation
from visualisation_tools import plot_tools
from launcher_tools import logger

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start an experiment')
    parser.add_argument('--n-disc', metavar='d', dest="n_disc", type=int, default=1,
                        help="Number of disc used in the experiment")
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
    parser.add_argument('--beta', dest="beta", type=float, default=10,
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
    parser.add_argument('--epoch-embedding', dest="epoch_embedding", type=int, default=500,
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
    parser.add_argument("--negative-sampling", dest="negative_sampling", type=int, default=5,
                        help="number of negative samples for loss O2")
    args = parser.parse_args()


    dataset_dict = { "karate": corpora.load_karate,
                "football": corpora.load_football,
                "flickr": corpora.load_flickr,
                "dblp": corpora.load_dblp,
                "books": corpora.load_books,
                "blogCatalog": corpora.load_blogCatalog
              }


    if(args.save):
        print("The following options are use for the current experiment ", args)
        os.makedirs("RESULTS/"+args.id+"/", exist_ok=True)
        logger_object = logger.JSONLogger("RESULTS/"+args.id+"/log.json")
        logger_object.append(vars(args))

    # check if dataset exists

    if(args.dataset not in dataset_dict):
        print("Dataset " + args.dataset + " does not exist, please select one of the following : ")
        print(list(dataset_dict.keys()))
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

    #X are edges, Y are labels, D are randomwalks

    print("Creating dataset")
    # index of examples dataset
    dataset_index = corpora_tools.from_indexable(torch.arange(0,len(D),1).unsqueeze(-1))


    print("Dataset Size -> ", len(D))
    D.set_path(False)

    # negative sampling distribution
    frequency = D.getFrequency()**(3/4)
    frequency[:,1] /= frequency[:,1].sum()
    frequency = pytorch_categorical.Categorical(frequency[:,1])
    # random walk dataset
    d_rw = D.light_copy()
    d_rw.set_walk(args.walk_lenght, 1.0)
    d_rw.set_path(True)
    d_rw = corpora.ContextCorpus(d_rw, context_size=args.context_size, precompute=args.precompute_rw)
    # neigbhor dataset
    d_v = D.light_copy()
    d_v.set_walk(1, 1.0)

    print(d_rw[1][0].size())

    print("Merging dataset")
    embedding_dataset = corpora_tools.zip_datasets(dataset_index,
                                                    corpora_tools.select_from_index(d_v, element_index=0),
                                                    d_rw
                                                    )
    training_dataloader = DataLoader(embedding_dataset,
                                batch_size=512,
                                shuffle=True,
                                num_workers=8,
                                collate_fn=data_tools.PadCollate(dim=0),
                                drop_last=False
                        )

    representation_d = []
    pi_d = []
    mu_d = []
    sigma_d = []

    for disc in range(args.n_disc):
        alpha, beta = args.init_alpha, args.init_beta
        embedding_alg = PEmbed(len(embedding_dataset), lr=args.init_lr, cuda=args.cuda, negative_distribution=frequency)
        em_alg = PEM(args.n_gaussian, init_mod="kmeans-hyperbolic", verbose=False)
        pi, mu, sigma = None, None, None
        pik = None
        for i in tqdm.trange(args.epoch):
            if(i==1):
                embedding_alg.set_lr(args.lr)
                alpha, beta = args.alpha, args.beta
            embedding_alg.fit(training_dataloader, alpha=alpha, beta=beta, gamma=args.gamma, max_iter=args.epoch_embedding,
                             pi=pik, mu=mu, sigma=sigma, negative_sampling=args.negative_sampling)
            em_alg.fit(embedding_alg.get_PoincareEmbeddings().cpu(), max_iter=5)
            pi, mu, sigma = em_alg.getParameters()
            pik = em_alg.getPik(embedding_alg.get_PoincareEmbeddings().cpu())
        representation_d.append(embedding_alg.get_PoincareEmbeddings().cpu())
        pi_d.append(pi)
        mu_d.append(mu)
        sigma_d.append(sigma)
        current_accuracy = evaluation.accuracy_cross_validation(representation_d[-1], D.Y, pi, mu, sigma, 5, verbose=False)
        print("\nPerformances disc "+str(disc+1)+"-> " ,current_accuracy,"\n")
        if(args.save):
            logger_object.append({"disc-"+str(disc):{"accuracy": current_accuracy}})

    #evaluate performances on all disc
    total_accuracy = evaluation.accuracy_disc_product(representation_d, D.Y, pi_d, mu_d, sigma_d, verbose=False)
    print("\nPerformances joined -> " ,
        total_accuracy
    )
    logger_object.append({"accuracy": total_accuracy})
    #TODO: Clean the code below

    if(args.save):
        import matplotlib.pyplot as plt
        import matplotlib.colors as plt_colors
        import numpy as np
        unique_label = np.unique(sum([ y for k, y in D.Y.items()],[]))
        colors = []

        for i in range(len(representation_d[0])):
            colors.append(plt_colors.hsv_to_rgb([D.Y[i][0]/(len(unique_label)),0.5,0.8]))



        plot_tools.plot_embedding_distribution_multi(representation_d, pi_d, mu_d,  sigma_d,
                                                    labels=None, N=100, colors=colors,
                                                    save_path="RESULTS/"+args.id+"/fig.pdf")

        torch.save(representation_d, "RESULTS/"+args.id+"/embeddings.t7")
        torch.save( {"pi": pi_d, "mu":mu_d, "sigma":sigma_d}, "RESULTS/"+args.id+"/pi_mu_sigma.t7")
