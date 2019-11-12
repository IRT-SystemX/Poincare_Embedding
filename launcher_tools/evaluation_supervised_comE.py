
import torch
import argparse
import tqdm
import io
import os


from em_tools.euclidean_em import EuclideanEM, GaussianMixtureSKLearn
from em_tools.euclidean_kmeans import KMeans
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
# representations /= 30
ground_truth = torch.LongTensor([[ 1 if(y+1 in Y[i]) else 0 for y in range(5)] for i in range(len(X))]).float()
print("gt", ground_truth.sum(0))
gg = ground_truth[:,0].nonzero()
print(gg)
rep = representations[gg.squeeze()]
print(rep.size())
x = ((rep.mean(0, keepdim=True).expand_as(rep) -  rep)**2).sum(-1)
N,M,D = ground_truth.size(0), ground_truth.size(1), representations.size(-1)
mean = (ground_truth.unsqueeze(-1).expand(N,M,D) * representations.unsqueeze(1).expand(N,M,D)).sum(0)/ground_truth.sum(0).unsqueeze(-1).expand(M,D)
distance_to_means = ((mean.unsqueeze(0).expand(N, M, D) - representations.unsqueeze(1).expand(N,M,D)) **2).sum(-1)
p_var = distance_to_means * ground_truth
print(p_var.mean(0))
print(p_var.sum(0)/ground_truth.sum(0))
print((((representations.mean(0,keepdim=True).expand(N,D) - representations)**2).sum(-1)).mean())
# print("Mean cluster 0 ", x.mean())
# print("All var ", representations.var())
# print(ground_truth.size())
# a = ground_truth.unsqueeze(-1).expand(N,M,D) * representations.unsqueeze(1).expand(N,M,D)
# f = ((a.mean(0).unsqueeze(0).expand(N,M,D) - a)**2).sum(-1)
# print("All var 2 ", f.mean(0))

# a = ground_truth.unsqueeze(-1).expand(N,M,D) * representations.unsqueeze(1).expand(N,M,D)
# print(a.sum(0).size())
# print(ground_truth.sum(0).size())
# f = (((a.sum(0)* (1/ground_truth.sum(0).unsqueeze(-1).expand(M,D))).unsqueeze(0).expand(N,M,D) - a)**2).sum(-1)
# print("All var 3 ", f.sum(0)/ground_truth.sum(0))

CVE = evaluation.CrossValEvaluation(representations, ground_truth, nb_set=5, algs_object=GaussianMixtureSKLearn)
scores = CVE.get_score(evaluation.PrecisionScore(at=1))

print("Mean score on the dataset -> ",sum(scores,0)/5)

log_in.append({"supervised_evaluation_em":scores})

CVE = evaluation.CrossValEvaluation(representations, ground_truth, nb_set=5, algs_object=KMeans)
scores = CVE.get_score(evaluation.PrecisionScore(at=1))

print("Mean score on the dataset -> ",sum(scores,0)/5)

log_in.append({"supervised_evaluation_kmeans":scores})