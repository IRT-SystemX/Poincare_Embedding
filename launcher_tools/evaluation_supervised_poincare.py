import argparse
import tqdm
import torch
import os

from torch.utils.data import DataLoader
from em_tools.poincare_em import RiemannianEM as EM
from em_tools.poincare_kmeans import PoincareKMeans as KM
from data_tools import corpora_tools, corpora, data, logger
from evaluation_tools import evaluation
from community_tools import poincare_classifier as pc


parser = argparse.ArgumentParser(description='Load embeddings and perform kmeans on it')

parser.add_argument('--file', dest="file", type=str, default="/local/gerald/POINCARE-EM/DT/dblp-2D-EM-TEST-13",
                    help="embeddings location file") 
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
size = log_in["size"]
if(dataset_name not in dataset_dict):
    print("Dataset " + dataset_name + " does not exist, please select one of the following : ")
    print(list(dataset_dict.keys()))
    quit()

print("Loading Corpus ")
D, X, Y = dataset_dict[dataset_name]()

if(args.init):
  print("init embedding")
  representations = torch.load(os.path.join(args.file,"embeddings_init.t7"))
else:
  representations = torch.load(os.path.join(args.file,"embeddings.t7"))[0]
print("rep -> ", representations.size())
ground_truth = torch.LongTensor([[ 1 if(y+1 in Y[i]) else 0 for y in range(n_gaussian)] for i in range(len(X))])
print("Ground truth size ", ground_truth.size())
print("##########################GMM Hyperbolic###############################")
CVE = evaluation.CrossValEvaluation(representations, ground_truth, nb_set=5, algs_object=EM)
p1 = CVE.get_score(evaluation.PrecisionScore(at=1))

p3 = CVE.get_score(evaluation.PrecisionScore(at=3))

p5 = CVE.get_score(evaluation.PrecisionScore(at=5))
scores = {"P1":p1, "P3":p3, "P5":p5}

print("\n\t score ->  ",{"P1":sum(p1,0)/5, "P3":sum(p3,0)/5, "P5":sum(p5,0)/5}, "\n\n")

log_in.append({"supervised_evaluation_em":scores})

CVE = evaluation.CrossValEvaluation(representations, ground_truth, nb_set=5, algs_object=KM)
scores = CVE.get_score(evaluation.PrecisionScore(at=1))

print("Mean score on the dataset kmean -> ",sum(scores,0)/5)

log_in.append({"supervised_evaluation_em":scores})