import argparse
import tqdm

import torch
import pytorch_categorical
from torch.utils.data import DataLoader
import os
from multiprocessing import Process, Manager
from em_tools.poincare_em import RiemannianEM as PEM
from data_tools import corpora_tools
from data_tools import corpora
from data_tools import data_tools
from evaluation_tools import evaluation
from visualisation_tools import plot_tools
from data_tools import logger
from optim_tools import optimizer

parser = argparse.ArgumentParser(description='Load embeddings and perform kmeans on it')

parser.add_argument('--file', dest="file", type=str, default="RESULTS/football-5D-KMEANS-1/",
                    help="embeddings location file")
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
            "adjnoun": corpora.load_adjnoun
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

representations = torch.load(os.path.join(args.file,"embeddings.t7"))[0]
# for i in tqdm.trange(args.n):
#     total_accuracy = evaluation.poincare_unsupervised_em(representations, D.Y, n_gaussian,  verbose=False)
#     results.append(total_accuracy)

from em_tools.poincare_em import RiemannianEM

em = RiemannianEM(representations.size(-1), n_gaussian, verbose=False)
em.fit(representations, max_iter=1)

import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import numpy as np

unique_label = np.unique(sum([ y for k, y in D.Y.items()],[]))
colors = []

for i in range(len(D.Y)):
    colors.append(plt_colors.hsv_to_rgb([D.Y[i][0]/(len(unique_label)),0.5,0.8]))
print(args.file)
print(os.path.join(args.file,"poincare_visualisation_pred.png"))
plot_tools.plot_embedding_distribution(representations, em._w, em._mu,  em._sigma, 
                                                labels=None, N=100, colors=colors, 
                                                save_path=os.path.join(args.file,"poincare_visualisation_pred.png"))

pred = em.predict(representations)
for i in range(len(D.Y)):
    colors.append(plt_colors.hsv_to_rgb([pred[i]/(len(unique_label)),0.5,0.8]))

plot_tools.plot_embedding_distribution(representations, em._w, em._mu,  em._sigma, 
                                                labels=None, N=100, colors=colors, 
                                                save_path=os.path.join(args.file,"poincare_visualisation_pred.png"))
