import io
import os
import torch 
import random
import tqdm
import time
from torch.utils.data import Dataset
from scipy import io as sio
from data_tools import dataset_downloader
from data_tools import data_tools
from torch.utils.data import DataLoader
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
    def getFrequency(self):
        return torch.Tensor([[k, len(v)] for k, v in self.X.items()])
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

class NeigbhorFlatCorpus(Dataset):
    def __init__(self, X, Y):
        # the sparse torch dictionary
        self.X = X
        self.Y = Y

        self.data = []
        for ns, nln in X.items():
            for nl in nln:
                self.data.append([ns, nln])

    def __getitem__(self, index):
        return torch.LongTensor([self.data[index][0]]), torch.LongTensor(self.data[index][1])

    def __len__(self):
        return len(self.data)

class ContextCorpus(Dataset):
    def __init__(self, dataset, context_size=5, precompute=-1):
        self._dataset = dataset
        self.c_s = context_size
        self.precompute = precompute
        if precompute > 0 :
            self.paths = self._precompute()

    def _precompute(self):
        precompute = self.precompute
        self.precompute = -1
        paths = []
        for i in tqdm.trange(len(self)):
            paths.append([self.__getitem__(i) for j in  range(precompute)])
        self.precompute = precompute
        print("sizes -> ", len(paths), len(paths[0]), len(paths[0][0][0]))
        print("npairs -> ", len(paths) * len(paths[0]) * len(paths[0][0][0]))
        return paths

    def __getitem__(self, index):
        if(self.precompute <= 0):
            path = self._dataset[index][0].squeeze()
            # print(path)
            x = [[path[i].item(), path[j].item()]  for i in range(len(path))
                    for j in range(max(0, i - self.c_s),min(len(path), i + self.c_s)) 
                    if(i!=j)]
            return (torch.LongTensor(x),)
        else:
            index_path = random.randint(0, self.precompute-1)
            return self.paths[index][index_path]

    def __len__(self):
        return len(self._dataset)

class ExtendedContextCorpus(Dataset):
    def __init__(self, dataset, context_size=5, precompute=1):
        self._dataset = dataset
        self.c_s = context_size
        self.precompute = precompute
        if(precompute < 1):
            print("Precompute is mandatory value "+str(precompute)+ " must be a positive integer instead")
            precompute = 1
        self.context = self._precompute()
        self.n_sample = 5

    def _precompute(self):
        precompute = self.precompute
        self.precompute = -1
        context = [set() for i in range(len(self._dataset))]
        for i in tqdm.trange(len(self._dataset)):
            # get the random walk
            path = self._dataset[i][0].squeeze()
            for k in range(len(path)):
                for j in range(max(0, k - self.c_s), min(len(path), k + self.c_s)):
                    if(k!=j):
                        context[path[k].item()].add(path[j].item())
        for i in range(len(context)):
            context[i] = torch.LongTensor(list(context[i]))

        return context

    def __getitem__(self, index):
        c_context = self.context[index]
        indexes = (torch.rand(self.n_sample) * len(c_context)).long()
        return (c_context[indexes],)

    def __len__(self):
        return len(self.context)

class FlatContextCorpus(Dataset):
    def __init__(self, dataset, context_size=5, precompute=1):
        self._dataset = dataset
        self.c_s = context_size
        self.precompute = precompute
        if(precompute < 1):
            print("Precompute is mandatory value "+str(precompute)+ " must be a positive integer instead")
            precompute = 1
        self.context = torch.LongTensor(self._precompute()).unsqueeze(-1)
        self.n_sample = 5
        # print("LEANANN ", len(self))

    def _precompute(self):
        precompute = self.precompute
        self.precompute = -1
        context = [[] for i in range(len(self._dataset))]
        for p in range(precompute):
            for i in tqdm.trange(len(self._dataset)):
                # get the random walk
                path = self._dataset[i][0].squeeze()
                for k in range(len(path)):
                    for j in range(max(0, k - self.c_s), min(len(path), k + self.c_s)):
                        if(k!=j):
                            context[path[k].item()].append(path[j].item())
        flat_context = []
        for i, v  in enumerate(context):
            for item in v:
                flat_context.append([i, item])
        
        return flat_context

    def __getitem__(self, index):
        a, b = self.context[index][0], self.context[index][1]
        return a, b
    def __len__(self):
        return len(self.context)


class WeightedFlatContextCorpus(Dataset):
    def __init__(self, dataset, context_size=5, precompute=1):
        self._dataset = dataset
        self.c_s = context_size
        self.precompute = precompute
        if(precompute < 1):
            print("Precompute is mandatory value "+str(precompute)+ " must be a positive integer instead")
            precompute = 1
        self.context = self._precompute()
        self.n_sample = 5
        print("LEANANN ", len(self))

    def _precompute(self):
        precompute = self.precompute
        self.precompute = -1
        context = [{} for i in range(len(self._dataset))]
        for p in range(precompute):
            for i in tqdm.trange(len(self._dataset)):
                # get the random walk
                path = self._dataset[i][0].squeeze()
                for k in range(len(path)):
                    for j in range(max(0, k - self.c_s), min(len(path), k + self.c_s)):
                        if(k!=j):
                            if(path[j].item() not in context[path[k].item()]):
                                context[path[k].item()][path[j].item()] = 0
                            context[path[k].item()][path[j].item()] += 1
        flat_context = []
        for i, v  in enumerate(context):
            for item, value in v.items():
                flat_context.append((torch.LongTensor([i]), torch.LongTensor([item]), torch.floatTensor([value])))
        return flat_context

    def __getitem__(self, index):
        return self.context[index]

    def __len__(self):
        return len(self.context)


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

# def loading_mat_txt(mat_path, label_path):
#     # Graph
#     X = {}
#     with io.open(mat_path, "r") as edges_file:
#         for i, line in enumerate(edges_file):
#             lsp = line.split()
#             X[i] = [k for k, value in enumerate(lsp) if(int(value) == 1)]
    
#     Y = {}
#     with io.open(label_path, "r") as label_file:
#         for i, line in enumerate(label_file):
            
#             Y[i] = []
#             Y[i].append(int(line))

#     return RandomWalkCorpus(X, Y), X, Y   

def load_dblp():
    # os.makedirs("data/DBLP/", exist_ok=True)
    # dataset_downloader.download("http://webia.lip6.fr/~gerald/data/graph/DBLP/Dblp.mat", "data/DBLP/Dblp.mat")
    # dataset_downloader.download("http://webia.lip6.fr/~gerald/data/graph/DBLP/labels.txt", "data/DBLP/labels.txt")
    mat_path = "data/DBLP/Dblp.mat"
    label_path = "data/DBLP/labels.txt"
    return loading_matlab_corpus(mat_path, label_path)

def load_flickr():
    # os.makedirs("data/FLICKR/", exist_ok=True)
    # dataset_downloader.download("http://webia.lip6.fr/~gerald/data/graph/Flickr-dataset/edges.csv", "data/FLICKR/edges.csv")
    # dataset_downloader.download("http://webia.lip6.fr/~gerald/data/graph/Flickr-dataset/group-edges.csv", "data/FLICKR/group-edges.csv")
    edges_path = "/local/gerald/data/FLICKR/edges.csv"
    groups_path = "/local/gerald/data/FLICKR/group-edges.csv"
    return loading_social_computing_corpus(edges_path, groups_path, symetric=True)

def load_blogCatalog():
    # os.makedirs("data/BlogCatalog-dataset/", exist_ok=True)
    # dataset_downloader.download("http://webia.lip6.fr/~gerald/data/graph/BlogCatalog-dataset/edges.csv", "data/BlogCatalog-dataset/edges.csv")
    # dataset_downloader.download("http://webia.lip6.fr/~gerald/data/graph/BlogCatalog-dataset/group-edges.csv", "data/BlogCatalog-dataset/group-edges.csv")
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

def load_adjnoun():
    matrix_path = "Input/Adjnoun.txt"
    label_path = "Input/R_Adjnoun.txt"
    return loading_mat_txt(matrix_path, label_path)

def load_polblogs():
    matrix_path = "Input/Polblogs.txt"
    label_path = "Input/R_Polblogs.txt"
    return loading_mat_txt(matrix_path, label_path)

def test_flat_context_corpus():
    dblp_dataset, X, Y = load_dblp()
    dblp_dataset.set_walk(5, 1.0)
    dblp_dataset.set_path(True)
    fcc = FlatContextCorpus(dblp_dataset, context_size=5, precompute=5)
    dataloader_slow = DataLoader(fcc, 
                                batch_size=2000, 
                                shuffle=True,
                                num_workers=2,
                                drop_last=False
                        )
    start_time = time.time()
    for i, *items in zip(tqdm.trange(len(dataloader_slow)), dataloader_slow):
        x = items[0]
        print(x)
    end_time = time.time()
    print("time to iterate all dataset", end_time-start_time)
    data_tools.RawDataloader(dataset, batch_size=2000)
    start_time = time.time()
    for i in tqdm.trange(len(fcc)//2000 +1):
        x = fcc[i*2000:max((i+1)*2000, len(fcc))]
        x += 1
    end_time = time.time()
    print("time to iterate all dataset", end_time-start_time)
test_flat_context_corpus()  