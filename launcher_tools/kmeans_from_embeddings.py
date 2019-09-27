import argparse
import tqdm

import torch
import pytorch_categorical
from torch.utils.data import DataLoader
import os
from multiprocessing import Process, Manager
from embedding_tools.poincare_embeddings_graph_multi import RiemannianEmbedding as PEmbed
from em_tools.poincare_em_multi import RiemannianEM as PEM
from data_tools import corpora_tools
from data_tools import corpora
from data_tools import data_tools
from evaluation_tools import evaluation
from visualisation_tools import plot_tools
from launcher_tools import logger
from optim_tools import optimizer

parser = argparse.ArgumentParser(description='Load embeddings and perform kmeans on it')

parser.add_argument('--file', dest="file", type=str, default="RESULTS/DBLP-3D-KMEANS-2/embeddings.t7",
                    help="embeddings location file")
parser.add_argument('--dataset', dest="dataset", type=str, default="dblp",
                    help="the dataset name")
parser.add_argument('--n', dest="n", type=int, default=1,
                    help="number of times to perform kmeans")      
parser.add_argument('--c', dest="c", type=int, default=5,
                    help="number of clusters")              
args = parser.parse_args()


dataset_dict = { "karate": corpora.load_karate,
            "football": corpora.load_football,
            "flickr": corpora.load_flickr,
            "dblp": corpora.load_dblp,
            "books": corpora.load_books,
            "blogCatalog": corpora.load_blogCatalog
          }
if(args.dataset not in dataset_dict):
    print("Dataset " + args.dataset + " does not exist, please select one of the following : ")
    print(list(dataset_dict.keys()))
    quit()

print("Loading Corpus ")
D, X, Y = dataset_dict[args.dataset]()

results = []
representations = torch.load(args.file)[0]

for i in tqdm.trange(args.n):
    total_accuracy = evaluation.accuracy_disc_kmeans(representations, D.Y, torch.zeros(args.c),  verbose=False)
    results.append(total_accuracy)

R = torch.Tensor(results)
print(results)
print("MEANS -> ", R.mean())
print("MAX -> ", R.max())
print("MIN -> ", R.min())
