import io
import os
import torch 
import random

from torch.utils.data import Dataset
from scipy import io as sio
from data_tools import dataset_downloader

class RandomWalkCorpus(Dataset):
    def __init__(self, X, Y, path=True):
        # the sparse torch dictionary
        self.X = X
        self.Y = Y
        self.k = 0
        self.path = path
        self.p_c = 1

    def set_walk(self, maximum_walk, continue_probability):
        self.k = maximum_walk
        self.p_c = continue_probability

    def set_path(self, path_val):
        self.path = path_val

    def light_copy(self):
        rwc_copy =  RandomWalkCorpus(self.X, self.Y, path=self.path)
        rwc_copy.k = self.k
        rwc_copy.p_c = self.p_c

        return rwc_copy

    def _walk(self, index):
        path = []
        c_index = index 
        path.append(c_index)
        for i in range(self.k):
            
            if(random.random()>self.p_c):
                break
            c_index = self.X[index][random.randint(0,len(self.X[index])-1)]
            path.append(c_index)
        return path if(self.path) else [c_index] 

    def __getitem__(self, index):
        return torch.LongTensor([self._walk(index)]), torch.LongTensor(self.Y[index])

    def __len__(self):
        return len(self.X)
def loading_matlab_corpus(mat_path, label_path):

    # Graph
    M = []
    with io.open(mat_path, "rb") as mat_file:
        M.append(sio.loadmat(mat_file)["network"])
    NNM_X, NNM_Y = M[0].nonzero()

    X = {}
    for i, (x, y) in enumerate(zip(NNM_X,NNM_Y)):
        if(x not in X):
            X[x] = []
        X[x].append(y)
    # Label
    Y = {}
    with io.open(label_path, "r") as label_file:
        for line in label_file:
            lsp = line.split()
            if(int(lsp[0])-1 not in Y):
                Y[int(lsp[0])-1] = []
            Y[int(lsp[0])-1].append(int(lsp[1]))

    return RandomWalkCorpus(X, Y), X, Y


def loading_social_computing_corpus(edges_path, groups_path, symetric=True):
    # Graph
    X = {}
    with io.open(edges_path, "r") as edges_file:
        for line in edges_file:
            lsp = line.split(",")
            if(int(lsp[0])-1 not in X):
                X[int(lsp[0])-1] = []
            X[int(lsp[0])-1].append(int(lsp[1])-1)
            if(symetric):
                if(int(lsp[1])-1 not in X):
                    X[int(lsp[1])-1] = []
                X[int(lsp[1])-1].append(int(lsp[0])-1)                
    # Label
    Y = {}
    with io.open(groups_path, "r") as label_file:
        for line in label_file:
            lsp = line.split(",")
            if(int(lsp[0])-1 not in Y):
                Y[int(lsp[0])-1] = []
            Y[int(lsp[0])-1].append(int(lsp[1]))
    # transform to tensor

    
    return RandomWalkCorpus(X, Y), X, Y    

def loading_mat_txt(mat_path, label_path):
    # Graph
    X = {}
    with io.open(mat_path, "r") as edges_file:
        for i, line in enumerate(edges_file):
            lsp = line.split()
            X[i] = [k for k, value in enumerate(lsp) if(int(value) == 1)]
    
    Y = {}
    with io.open(label_path, "r") as label_file:
        for i, line in enumerate(label_file):
            
            Y[i] = []
            Y[i].append(int(line))

    return RandomWalkCorpus(X, Y), X, Y   

def load_dblp():
    os.makedirs("data/DBLP/", exist_ok=True)
    dataset_downloader.download("http://webia.lip6.fr/~gerald/data/graph/DBLP/Dblp.mat", "data/DBLP/Dblp.mat")
    dataset_downloader.download("http://webia.lip6.fr/~gerald/data/graph/DBLP/labels.txt", "data/DBLP/labels.txt")
    mat_path = "data/DBLP/Dblp.mat"
    label_path = "data/DBLP/labels.txt"
    return loading_matlab_corpus(mat_path, label_path)

def load_flickr():
    os.makedirs("data/FLICKR/", exist_ok=True)
    dataset_downloader.download("http://webia.lip6.fr/~gerald/data/graph/Flickr-dataset/edges.csv", "data/FLICKR/edges.csv")
    dataset_downloader.download("http://webia.lip6.fr/~gerald/data/graph/Flickr-dataset/group-edges.csv", "data/FLICKR/group-edges.csv")
    edges_path = "data/FLICKR/edges.csv"
    groups_path = "data/FLICKR/group-edges.csv"
    return loading_social_computing_corpus(edges_path, groups_path, symetric=True)

def load_blogCatalog():
    os.makedirs("data/BlogCatalog-dataset/", exist_ok=True)
    dataset_downloader.download("http://webia.lip6.fr/~gerald/data/graph/BlogCatalog-dataset/edges.csv", "data/BlogCatalog-dataset/edges.csv")
    dataset_downloader.download("http://webia.lip6.fr/~gerald/data/graph/BlogCatalog-dataset/group-edges.csv", "data/BlogCatalog-dataset/group-edges.csv")
    edges_path = "data/BlogCatalog-dataset/edges.csv"
    groups_path = "data/BlogCatalog-dataset/group-edges.csv"
    return loading_social_computing_corpus(edges_path, groups_path, symetric=True)

def load_karate():
    matrix_path = "Input/Karate.txt"
    label_path = "Input/R_Karate.txt"
    return loading_mat_txt(matrix_path, label_path)

def load_books():
    matrix_path = "Input/Books.txt"
    label_path = "Input/R_Books.txt"
    return loading_mat_txt(matrix_path, label_path)

def load_football():
    matrix_path = "Input/Football.txt"
    label_path = "Input/R_Football.txt"
    return loading_mat_txt(matrix_path, label_path)