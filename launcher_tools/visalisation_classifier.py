import argparse
import tqdm
import torch
import os

from torch.utils.data import DataLoader
from clustering_tools.poincare_em import PoincareEM as EM
from data_tools import corpora_tools, corpora, data, logger
from evaluation_tools import evaluation
from community_tools import poincare_classifier as pc
from visualisation_tools import plot_tools

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
            "adjnoun": corpora.load_adjnoun
          }
log_in = logger.JSONLogger(os.path.join(args.file,"log.json"), mod="continue")
dataset_name = log_in["dataset"]
print(dataset_name)
n_gaussian = log_in["n_gaussian"]
if(dataset_name not in dataset_dict):
    print("Dataset " + dataset_name + " does not exist, please select one of the following : ")
    print(list(dataset_dict.keys()))
    quit()


print("Loading Corpus ")
D, X, Y = dataset_dict[dataset_name]()

results = []
std_kmeans = []
if(args.init):
  print("init embedding")
  representations = torch.load(os.path.join(args.file,"embeddings_init.t7"))
else:
  representations = torch.load(os.path.join(args.file,"embeddings.t7"))[0]
ground_truth = torch.LongTensor([[ 1 if(y+1 in Y[i]) else 0 for y in range(n_gaussian)] for i in range(len(X))])
kmeans = pc.PoincareClassifier(n_gaussian)
kmeans.fit(representations, ground_truth)
gt_colors = []
pr_colors = []

import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import numpy as np
unique_label = np.unique(sum([ y for k, y in D.Y.items()],[]))

prediction = kmeans.predict(representations)+1
print(prediction.size())
for i in range(len(D.Y)):
    gt_colors.append(plt_colors.hsv_to_rgb([D.Y[i][0]/(len(unique_label)),0.5,0.8]))
    pr_colors.append(plt_colors.hsv_to_rgb([prediction[i].item()/(len(unique_label)),0.5,0.8]))

plot_tools.plot_embeddings(representations,  colors=pr_colors, save_folder=args.file, file_name="pred_classifier.png")
plot_tools.plot_embeddings(representations,  colors=gt_colors, save_folder=args.file, file_name="ground_truth.png")

