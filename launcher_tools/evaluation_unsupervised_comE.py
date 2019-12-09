import argparse
import tqdm

import torch
import pytorch_categorical
from torch.utils.data import DataLoader
import os
from multiprocessing import Process, Manager
from clustering_tools.euclidean_em import GaussianMixtureSKLearn
from data_tools import corpora_tools
from data_tools import corpora
from data_tools import data
from evaluation_tools import evaluation
from visualisation_tools import plot_tools
from data_tools import logger
from optim_tools import optimizer

parser = argparse.ArgumentParser(description='Load embeddings and perform kmeans on it')


parser.add_argument('--file', dest="file", type=str, default="/local/gerald/ComE/DBLP-128-2",
                    help="embeddings location file") 
parser.add_argument('--dataset', dest="dataset", type=str, default="dblp",
                    help="dataset") 
parser.add_argument('--n-gaussian', dest="n_gaussian", type=int, default=5,
                    help="dataset") 
parser.add_argument('--n', dest="n", type=int, default=1,
                    help="number of times to perform kmeans")          


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
dataset_name = args.dataset
n_gaussian = args.n_gaussian

if(dataset_name not in dataset_dict):
    print("Dataset " + dataset_name + " does not exist, please select one of the following : ")
    print(list(dataset_dict.keys()))
    quit()


print("Loading Corpus ")

D, X, Y = dataset_dict[dataset_name]()
import io
with io.open(os.path.join(args.file, "embeddings.txt")) as embedding_file:
    V = []
    for line in embedding_file:
        splitted_line = line.split()
        V.append([float(splitted_line[i+1]) for i in range(len(splitted_line)-1)])


representations = torch.Tensor(V)

print(representations.size())
print(len(Y))
results = []

for i in tqdm.trange(args.n):
    total_accuracy = evaluation.euclidean_unsupervised_em(representations, D.Y, n_gaussian,  verbose=False)
    results.append(total_accuracy)

R = torch.Tensor(results)
print("Maximum performances -> ", R.max().item())
print("Mean performances -> ", R.mean().item())
print("std performances -> ", R.std().item())
log_in.append({"evaluation_unsupervised_come": {"unsupervised_performances":R.tolist()}})







# print(prediction_mat.sum(0))

conductences = []
adjency_matrix = X

for i in tqdm.trange(args.n):
    algs = GaussianMixtureSKLearn(5)
    algs.fit(representations)
    prediction = algs.predict(representations).long()
    # print(prediction)
    prediction_mat = torch.LongTensor([[ 1 if(y == prediction[i]) else 0 for y in range(5)] for i in range(len(X))])
    # print(prediction_mat)
    conductences.append(evaluation.mean_conductance(prediction_mat, adjency_matrix))

C = torch.Tensor(conductences)
print("Maximum conductence -> ", C.max().item())
print("Mean conductence -> ", C.mean().item())
print("stdconductence -> ", C.std().item())
log_in.append({"evaluation_unsupervised_come": {"unsupervised_conductence":C.tolist()}})


nmi = []
ground_truth = torch.LongTensor([[ 1 if(y in Y[i]) else 0 for y in range(5)] for i in range(len(X))])
for i in tqdm.trange(args.n):
    algs = GaussianMixtureSKLearn(5)
    algs.fit(representations)
    prediction = algs.predict(representations).long()
    prediction_mat = torch.LongTensor([[ 1 if(y == prediction[i]) else 0 for y in range(5)] for i in range(len(X))])
    nmi.append(evaluation.nmi(prediction_mat, ground_truth))

C = torch.Tensor(nmi)
print("Maximum nmi -> ", C.max().item())
print("Mean nmi -> ", C.mean().item())
print("std nmi -> ", C.std().item())
log_in.append({"evaluation_unsupervised_come_nmi": {"unsupervised_nmi":C.tolist()}})