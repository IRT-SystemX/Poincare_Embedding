
import torch
import argparse
import tqdm
import io
import os


from community_tools.euclidean_classifier import EuclideanClassifier, SKLearnSVM
from data_tools import corpora_tools, corpora, data, logger
from evaluation_tools import evaluation

parser = argparse.ArgumentParser(description='Load embeddings and perform kmeans on it')

parser.add_argument('--file', dest="file", type=str, default="/local/gerald/ComE/DBLP-128-2",
                    help="embeddings location file") 
parser.add_argument('--dataset', dest="dataset", type=str, default="dblp",
                    help="dataset") 
parser.add_argument('--n-gaussian', dest="n_gaussian", type=int, default=5,
                    help="dataset") 
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

# getting paramaeters
dataset_name = args.dataset
n_gaussian = args.n_gaussian

# check if dataset exists
if(dataset_name not in dataset_dict):
    print("Dataset " + dataset_name + " does not exist, please select one of the following : ")
    print(list(dataset_dict.keys()))
    quit()


# loading the corpus
print("Loading Corpus ")
D, X, Y = dataset_dict[dataset_name]()


with io.open(os.path.join(args.file, "embeddings.txt")) as embedding_file:
    X = []
    for line in embedding_file:
        splitted_line = line.split()
        X.append([float(splitted_line[i+1]) for i in range(len(splitted_line)-1)])


representations = torch.Tensor(X)
ground_truth = torch.LongTensor([[ 1 if(y+1 in Y[i]) else 0 for y in range(n_gaussian)] for i in range(len(X))])

# learning a simple classifier
from torch import nn


CVE = evaluation.CrossValEvaluation(representations, ground_truth, nb_set=5, algs_object=SKLearnSVM)
scores = CVE.get_score(evaluation.PrecisionScore(at=1))

print("Mean score on the dataset -> ",sum(scores,0)/5)

log_in.append({"supervised_evaluation_SKLearnSVM":scores})

CVE = evaluation.CrossValEvaluation(representations, ground_truth, nb_set=5, algs_object=EuclideanClassifier)
p1 = CVE.get_score(evaluation.PrecisionScore(at=1))

p3 = CVE.get_score(evaluation.PrecisionScore(at=3))

p5 = CVE.get_score(evaluation.PrecisionScore(at=5))
scores = {"P1":p1, "P3":p3, "P5":p5}

print("\n\t score ->  ",{"P1":sum(p1,0)/5, "P3":sum(p3,0)/5, "P5":sum(p5,0)/5}, "\n\n")
log_in.append({"supervised_evaluation_EuclideanClassifier":scores})