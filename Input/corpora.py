import io
import torch 
import random

from torch.utils.data import Dataset
from scipy import io as sio


class DeepWalkCorpus(Dataset):
    def __init__(self, X, Y, path=True):
        # the sparse torch dictionary
        self.X = X
        self.Y = Y
        self.k = 0
        self.path = path

    def set_walk(self, maximum_walk, continue_probability):
        self.k = maximum_walk
        self.p_c = continue_probability

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
        return torch.LongTensor(self._walk(index)), torch.LongTensor(self.Y[index])

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
            Y[int(lsp[0])-1] = int(lsp[1])

    return DeepWalkCorpus(X, Y), X, Y


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

    return DeepWalkCorpus(X, Y), X, Y    

def test_dplp():
    mat_path = "/local/gerald/data/Communauty/DBLP/Dblp.mat"
    label_path = "/local/gerald/data/Communauty/DBLP/labels.txt"
    return loading_matlab_corpus(mat_path, label_path)

def test_flickr():
    edges_path = "/local/gerald/data/Communauty/Flickr-dataset/edges.csv"
    groups_path = "/local/gerald/data/Communauty/Flickr-dataset/group-edges.csv"
    return loading_social_computing_corpus(edges_path, groups_path, symetric=True)