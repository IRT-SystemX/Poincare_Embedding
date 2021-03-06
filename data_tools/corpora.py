import io
import os
import torch 
import random
import tqdm
import time
from torch.utils.data import Dataset
from scipy import io as sio
from data_tools import dataset_downloader
from data_tools import data as dts
from torch.utils.data import DataLoader
from joblib import Parallel, delayed

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

class FixedRandomWalkCorpus(RandomWalkCorpus):
    def __init__(self, X, Y, path=True, precompute=1, path_len=10):
        super(FixedRandomWalkCorpus, self).__init__(X, Y, path=path)
        self.precompute = precompute
        self.k = path_len
        self.p_c = 1
        self.paths = self._precompute()

    def _precompute(self):
        precompute = self.precompute
        self.precompute = -1
        paths = []
        for index in tqdm.trange(len(self.X)):
            for i in range(precompute):
                    paths.append(torch.LongTensor(self._walk(index)).unsqueeze(0))
        self.precompute = precompute
        return torch.cat(paths,0)


    def cuda(self):
        self.paths = self.paths.cuda()

    def cpu(self):
        self.paths = self.paths.cpu()

    def __getitem__(self, index):
        return self.paths[index]
        
    def __len__(self):
        return len(self.paths)

class NeigbhorFlatCorpus(Dataset):
    def __init__(self, X, Y):
        # the sparse torch dictionary
        self.X = X
        self.Y = Y

        self.data = []
        for ns, nln in X.items():
            for nl in nln:
                self.data.append([ns, nl])
        self.data = torch.LongTensor(self.data)

    def cuda(self):
        self.data = self.data.cuda()

    def cpu(self):
        self.data = self.data.cpu()

    def __getitem__(self, index):
        if(type(index) == int):
            a, b = self.data[index][0], self.data[index][1]
        else:
            a,b  = self.data[index][:,0], self.data[index][:,1]
        return a, b
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

    def cuda(self):
        print("max index ", self.context.max())
        self.context = self.context.cuda()

    def cpu(self):
        self.context = self.context.cpu()

    def __getitem__(self, index):
        if(type(index) == int):
            a, b = self.context[index][0], self.context[index][1]
        else:
            a,b  = self.context[index][:,0], self.context[index][:,1]
        return a, b
    def __len__(self):
        return len(self.context)


class RandomContextSize(RandomWalkCorpus):
    def __init__(self, X, Y, path=True, precompute=1, path_len=10, context_size=5):
        super(RandomContextSize, self).__init__(X, Y, path=path)
        self.precompute = precompute
        self.k = path_len
        self.p_c = 1
        self.c_s = context_size
        self.paths = self._precompute()

    def _precompute(self):
        precompute = self.precompute
        self.precompute = -1
        paths = []
        for index in tqdm.trange(len(self.X)):
            for i in range(precompute):
                    paths.append(torch.LongTensor(self._walk(index)).unsqueeze(0))
        self.precompute = precompute
        return torch.cat(paths,0)


    def cuda(self):
        self.paths = self.paths.cuda()

    def cpu(self):
        self.paths = self.paths.cpu()

    @staticmethod
    def _context_calc(k, path, max_context):
        context_size = random.randint(1, max_context -1)
        v = torch.cat((path[max(0, k - context_size):k],path[k+1: min(len(path), k + context_size)]) ,0)
        v =  v.unsqueeze(-1).expand(v.size(0), 2).clone()
        v[:,0] = path[k]
        return v

    def __getitem__(self, index):
        path = self.paths[index]

        res = [RandomContextSize._context_calc(i, path, self.c_s) for i in range(len(path))]

        return torch.cat(res, 0)

    def __len__(self):
        return len(self.paths)

class RandomContextSizeFlat(RandomWalkCorpus):
    def __init__(self, X, Y, path=True, precompute=1, path_len=10, context_size=5):
        super(RandomContextSizeFlat, self).__init__(X, Y, path=path)
        self.precompute = precompute
        self.k = path_len
        self.p_c = 1
        self.c_s = context_size
        self.paths = self._precompute()

    def _precompute(self):
        precompute = self.precompute
        self.precompute = -1
        paths = []
        for index in tqdm.trange(len(self.X)):
            for i in range(precompute):
                    paths.append(torch.LongTensor(self._walk(index)).unsqueeze(0))
        self.precompute = precompute
        return torch.cat(paths,0)


    def cuda(self):
        self.paths = self.paths.cuda()

    def cpu(self):
        self.paths = self.paths.cpu()

    @staticmethod
    def _context_calc(k, path, max_context):
        context_size = random.randint(1, max_context -1)
        v = torch.cat((path[max(0, k - context_size):k],path[k+1: min(len(path), k + context_size)]) ,0)
        v =  v.unsqueeze(-1).expand(v.size(0), 2).clone()
        v[:,0] = path[k]
        return v

    def __getitem__(self, index):

        path = self.paths[index//self.k]

        res = RandomContextSize._context_calc(index%self.k, path,self.c_s) 
        return res

    def __len__(self):
        return len(self.paths) * self.k



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

def saving_matlab_corpus(X, Y, filepath_mat, filepath_label):
    from scipy.sparse import csr_matrix
    from scipy.io import savemat
    
    row = []
    col = []
    val = []
    for x, neigbhor_x in X.items():
        for n in neigbhor_x:
            row.append(x)
            col.append(n)
            val.append(1)

    sm = csr_matrix((val, (row,col)), shape=(len(X), len(X)))  
    savemat(filepath_mat, {"network":sm})
    with io.open(filepath_label, "w") as label_file:
        for index, labels in Y.items():
            for label in labels:
                label_file.write(str(int(index+1))+"\t"+str(int(label))+"\n")

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
            print(int(line))

    return RandomWalkCorpus(X, Y), X, Y   




def load_dblp():

    mat_path = "data/DBLP/Dblp.mat"
    label_path = "data/DBLP/labels.txt"
    return loading_matlab_corpus(mat_path, label_path)

def load_wikipedia():
    mat_path  = "data/wikipedia/wikipedia.mat"
    label_path = "data/wikipedia/wikipedia.labels"
    return loading_matlab_corpus(mat_path, label_path)


def load_flickr():

    edges_path = "data/Flickr/edges.csv"
    groups_path = "data/Flickr/group-edges.csv"
    return loading_social_computing_corpus(edges_path, groups_path, symetric=True)

def load_blogCatalog():

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

def test():
    dblp_dataset, X, Y = load_dblp()
    dblp_dataset.set_walk(3, 1.0)
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
        # print(x[0].size())
        g = x[0] + 1
    end_time = time.time()
    print("time to iterate all dataset", end_time-start_time)
    dataloader_fast = dts.RawDataloader(fcc, batch_size=2000)
    start_time = time.time()
    print(fcc[0:2000][0].size())
    for i, *items in zip(tqdm.trange(len(dataloader_fast)), dataloader_fast):
        x = items[0]

        g = x[0]+1
    end_time = time.time()
    print("time to iterate all dataset", end_time-start_time)

if __name__ == '__main__':
    test()