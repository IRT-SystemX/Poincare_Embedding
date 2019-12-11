import argparse
import tqdm

import torch
import pytorch_categorical
from torch.utils.data import DataLoader
import os
from multiprocessing import Process, Manager
from clustering_tools.poincare_em import PoincareEM
from data_tools import corpora_tools
from data_tools import corpora
from data_tools import data
from evaluation_tools import evaluation
from visualisation_tools import plot_tools
from data_tools import logger
from optim_tools import optimizer

parser = argparse.ArgumentParser(description='Load embeddings and perform kmeans on it')

parser.add_argument('--file', dest="file", type=str, default="RESULTS/football-5D-KMEANS-1/",
                    help="embeddings location file")
parser.add_argument('--n', dest="n", type=int, default=1,
                    help="number of times to perform kmeans")          
parser.add_argument('--init', dest="init", action="store_true") 

args = parser.parse_args()


dataset_dict = { "karate": corpora.load_karate,
            "football": corpora.load_football,
            "flickr": corpora.load_flickr,
            "dblp": corpora.load_dblp,
            "books": corpora.load_books,
            "blogCatalog": corpora.load_blogCatalog,
            "polblog": corpora.load_polblogs,
            "adjnoun": corpora.load_adjnoun,
            "wikipedia": corpora.load_wikipedia
          }
log_in = logger.JSONLogger(os.path.join(args.file,"log.json"), mod="continue")
dataset_name = log_in["dataset"]
print(dataset_name)
n_gaussian = log_in["n_gaussian"]
dime = log_in["size"]
if(dataset_name not in dataset_dict):
    print("Dataset " + dataset_name + " does not exist, please select one of the following : ")
    print(list(dataset_dict.keys()))
    quit()


print("Loading Corpus ")
D, X, Y = dataset_dict[dataset_name]()

results = []

if(args.init):
  print("init embedding")
  representations = torch.load(os.path.join(args.file,"embeddings_init.t7"))
else:
  representations = torch.load(os.path.join(args.file,"embeddings.t7"))[0]

for i in tqdm.trange(args.n):
    print("Number of gaussian ", n_gaussian)
    total_accuracy = evaluation.poincare_unsupervised_em(representations, D.Y, n_gaussian,  verbose=False)
    results.append(total_accuracy)

R = torch.Tensor(results)
print("Maximum performances -> ", R.max().item())
print("Mean performances -> ", R.mean().item())
print("STD performances -> ", R.std().item())
log_in.append({"evaluation_unsupervised_poincare_precision": {"unsupervised_performances":R.tolist()}})







# print(prediction_mat.sum(0))

conductences = []
adjency_matrix = X
for i in tqdm.trange(args.n):
    algs = PoincareEM(n_gaussian)
    algs.fit(representations)
    prediction = algs.predict(representations)
    prediction_mat = torch.LongTensor([[ 1 if(y == prediction[i]) else 0 for y in range(n_gaussian)] for i in range(len(X))])
    conductences.append(evaluation.mean_conductance(prediction_mat, adjency_matrix))

C = torch.Tensor(conductences)
print("Maximum conductence -> ", C.max().item())
print("Mean conductence -> ", C.mean().item())
print("Mean conductence -> ", C.std().item())
log_in.append({"evaluation_unsupervised_poincare_conductance": {"unsupervised_conductence":C.tolist()}})

nmi = []
ground_truth = torch.LongTensor([[ 1 if(y in Y[i]) else 0 for y in range(n_gaussian)] for i in range(len(X))])
for i in tqdm.trange(args.n):
    algs = PoincareEM(n_gaussian)
    algs.fit(representations)
    prediction = algs.predict(representations)
    prediction_mat = torch.LongTensor([[ 1 if(y == prediction[i]) else 0 for y in range(n_gaussian)] for i in range(len(X))])
    nmi.append(evaluation.nmi(prediction_mat, ground_truth))

C = torch.Tensor(nmi)
print("Maximum nmi -> ", C.max().item())
print("Mean nmi -> ", C.mean().item())
print("Mean nmi -> ", C.std().item())
log_in.append({"evaluation_unsupervised_poincare_nmi": {"unsupervised_nmi":C.tolist()}})