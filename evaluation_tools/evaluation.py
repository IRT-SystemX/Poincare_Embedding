import torch
from function_tools import distribution_function, poincare_function
from function_tools import euclidean_function as ef

from em_tools.poincare_kmeans import PoincareKMeans
from em_tools.poincare_em import RiemannianEM
from collections import Counter
import numpy as np
import math
import tqdm
import itertools

# verified 
class EvaluationMetrics(object):
    # both must be matrix 
    def __init__(self, prediction, ground_truth, nb_classes):
        self.TP = []
        self.FP = []
        self.TN = []
        self.FN = []

        for i in range(nb_classes):
            positive_gt = (ground_truth[:,i] == 1).float()
            positive_pr = (prediction[:,i] == 1).float()
            tp = (positive_gt * positive_pr).sum()
            fp = positive_gt.sum() - tp
            self.TP.append(tp)
            self.FP.append(fp)

            negative_gt = (ground_truth[:,i] == 0).float()
            negative_pr = (prediction[:,i] == 0).float()
            tn = (negative_gt * negative_pr).sum()
            fn = negative_gt.sum() - tn
            self.TN.append(tn)
            self.FN.append(fn)            
    def micro_precision(self):
        return sum(self.TP, 0)/(sum(self.TP, 0) + sum(self.FP,0))

    def micro_recall(self):
        return sum(self.TP, 0)/(sum(self.TP, 0) + sum(self.FN,0))
    
    def micro_F(self):
        m_p, m_r  = self.micro_precision(), self.micro_recall()
        return (2 * m_p * m_r) /(m_p + m_r)

    def macro_precision(self):
        precision_by_label = [tp/(tp+fp) for tp, fp in zip(self.TP, self.FP)]
        return sum(precision_by_label, 0)/(len(precision_by_label))

    def macro_recall(self):
        recall_by_label = [tp/(tp+fn) for tp, fn in zip(self.TP, self.FN)]
        return sum(recall_by_label, 0)/(len(recall_by_label))

    def macro_F(self):
        m_p, m_r  = self.macro_precision(), self.macro_recall()
        return (2 * m_p * m_r) /(m_p + m_r)

    def score(self):
        return self.micro_F(), self.macro_F()

def precision_at(prediction, ground_truth, at=5):
    prediction_value, prediction_index = (-prediction).sort(-1)
    trange = torch.arange(len(prediction)).unsqueeze(-1).expand(len(prediction), at).flatten()
    indexes = prediction_index[:,:at].flatten()
    score = ((ground_truth[trange, indexes]).float().view(len(prediction), at)).sum(-1)/at
    return score.mean().item()


def mean_conductance(prediction, adjency_matrix):
    N = prediction.size(0)
    # the number of clusters
    K = prediction.size(-1)
    # print(K)
    I = {i for i in range(len(prediction))}

    score = 0
    for c in range(K):
        # print(prediction[:, c].nonzero().flatten())
        c_nodes = set(prediction[:, c].nonzero().flatten().tolist())
        nc_nodes = I - c_nodes
        cut_score_a = 0
        for i in c_nodes:
            cut_score_a += len(set(adjency_matrix[i]) - c_nodes)
            # for j in nc_nodes:
            #     if(j in adjency_matrix[i]):
            #         cut_score_a += 1
        cut_score_b = 0
        for i in c_nodes:
            cut_score_b += len(adjency_matrix[i])

        cut_score_c = 0
        for i in nc_nodes:
            cut_score_c += len(adjency_matrix[i])
        if(cut_score_b==0 or cut_score_c ==0):
            score += 0 
        else:
            score += cut_score_a/(min(cut_score_b, cut_score_c))
    
    return score/K


def nmi(prediction, ground_truth):
    N = prediction.size(0)
    # the number of clusters
    K = prediction.size(-1)

    I = {i for i in range(len(prediction))}

    PN = []
    GN = []
    den = 0
    for i in range(K):
        PN.append(set(prediction[:, i].nonzero().flatten().tolist()))
        GN.append(set(ground_truth[:, i].nonzero().flatten().tolist()))
        if(len(PN[-1]) != 0):
            den += len(PN[-1]) * math.log(len(PN[-1])/N)
        if(len(GN[-1]) != 0):
            den += len(GN[-1]) * math.log(len(GN[-1])/N)
    num = 0
    for a in PN:
        for b in GN:
            N_ij = len(a.intersection(b))
            if(N_ij != 0):
                num += N_ij * math.log((N_ij * N)/(len(a) *len(b) ))
    
    return -2 * (num/den)

class PrecisionScore(object):
    def __init__(self, at=5):
        self.at = at

    def __call__(self, x, y):
        return precision_at(x, y, at=self.at)




class CrossValEvaluation(object):
    def __init__(self, embeddings, ground_truth, nb_set=5, algs_object=RiemannianEM):
        self.algs_object = algs_object
        self.z = embeddings
        self.gt = ground_truth
        self.nb_set = nb_set
        # split set
        subset_index = torch.randperm(len(self.z))
        nb_value = len(self.z)//nb_set
        self.subset_indexer = [subset_index[nb_value *i:min(nb_value * (i+1), len(self.z))] for i in range(nb_set)]
        self.all_algs = []
        pb = tqdm.trange(len(self.subset_indexer))

        for i, test_index in zip(pb, self.subset_indexer):
            # create train dataset being concatenation of not current test set
            train_index = torch.cat([ subset for ci, subset in enumerate(self.subset_indexer) if(i!=ci)], 0)
            
            # get embeddings sets
            train_embeddings = self.z[train_index]
            test_embeddings  = self.z[test_index]

            # get ground truth sets
            train_labels = self.gt[train_index]
            test_labels  =  self.gt[test_index]


            algs = self.algs_object(self.gt.size(-1))
            algs.fit(train_embeddings, Y=train_labels)
            self.all_algs.append(algs)
        
    def get_score(self, scoring_function):
        scores = []
        pb = tqdm.trange(len(self.subset_indexer))
        for i, test_index in zip(pb, self.subset_indexer):
            # create train dataset being concatenation of not current test set
            train_index = torch.cat([ subset for ci, subset in enumerate(self.subset_indexer) if(i!=ci)], 0)
            
            # get embeddings sets
            train_embeddings = self.z[train_index]
            test_embeddings  = self.z[test_index]

            # get ground truth sets
            train_labels = self.gt[train_index]
            test_labels  =  self.gt[test_index]

            # must give the matrix of scores
            # print(algs._w)
            prediction = self.all_algs[i].probs(test_embeddings)
            # print(prediction.mean(0))
            # print("Pred size ", prediction.size())
            # print("Test size ", test_labels.size())
            set_score = scoring_function(prediction, test_labels)
            scores.append(set_score)
        return scores




def accuracy(prediction, labels):
    return (prediction == labels).float().mean()

########################################### TO CLEAN ##############################################
def predict(Z_train, Z_test, Y_train, Y_test, pi, mu, sigma):
    
    G_train = distribution_function.weighted_gmm_pdf(pi, Z_train, mu, sigma, poincare_function.distance)
    G_train = G_train.max(-1)[1]+1
    
    # for each class we count
    predict_class = torch.zeros(len(mu), len(pi))
    for j, v in enumerate(G_train):
        predict_class[v.item()-1][torch.LongTensor(Y_train[j])-1] +=1 
    sv, si = predict_class.sort(-1)
    g = torch.zeros(len(mu))
    for k in range(len(pi)):
        clas = torch.argmax(predict_class,-1)
        gaus = predict_class[torch.arange(0,len(predict_class)),clas].argmax()
        clas = clas[gaus]
        predict_class[gaus] = -1
        #predict_class[:,clas] = -1
        g[gaus] = clas
    
    # predict
    G_test= distribution_function.weighted_gmm_pdf(pi, Z_test, mu, sigma, poincare_function.distance)
    G_test= G_test.max(-1)[1]+1        

    prediction = g[G_test-1].long()
    return prediction

def accuracy_cross_validation_multi_disc(Z, Y, pi, mu, sigma, nb_set, verbose=True):
    subset_index = torch.randperm(len(Z[0]))
    nb_value = len(Z[0])//nb_set
    I_CV = [subset_index[nb_value *i:min(nb_value * (i+1), len(Z[0]))] for i in range(nb_set)]
    acc_total = 0.
    for i, test_index in enumerate(I_CV):
        # create train dataset
        train_index = torch.cat([ subset for ci, subset in enumerate(I_CV) if(i!=ci)],0)
        Y_train = [Y[ic.item()] for ic in train_index]

        #create test datase

        Y_test = [Y[ic.item()] for ic in test_index]        
        
        if(verbose):
            print("Set "+str(i)+" :")
            print("\t train size -> "+str(len(Y_train)))
            print("\t test size -> "+str(len(Y_test)))
            print("Associate to each gaussian a class")
        predictions = []
        for j in range(len(Z)):
            Z_train = Z[j][train_index]
            Z_test = Z[j][test_index]
            predictions.append(predict(Z_train, Z_test, Y_train, Y_test, pi[j], mu[j], sigma[j]).unsqueeze(-1))
        predictions = torch.cat(predictions, -1)
        predictions = predictions.tolist()
        prediction = torch.LongTensor([Counter(l).most_common()[0][0] for l in predictions])
        acc = accuracy(prediction, torch.LongTensor([i[0]-1 for i in Y_test]))
        acc_total += acc.item()
    return acc_total/(len(I_CV))


def accuracy_cross_validation(Z, Y, pi,  mu, sigma, nb_set, verbose=True):
    subset_index = torch.randperm(len(Z))
    nb_value = len(Z)//nb_set
    I_CV = [subset_index[nb_value *i:min(nb_value * (i+1), len(Z))] for i in range(nb_set)]
    acc_total = 0.
    for i, test_index in enumerate(I_CV):
        # create train dataset
        train_index = torch.cat([ subset for ci, subset in enumerate(I_CV) if(i!=ci)],0)
        Z_train = Z[train_index]
        Y_train = [Y[ic.item()] for ic in train_index]

        #create test datase
        Z_test = Z[test_index]
        Y_test = [Y[ic.item()] for ic in test_index]        
        
        if(verbose):
            print("Set "+str(i)+" :")
            print("\t train size -> "+str(len(Z_train)))
            print("\t test size -> "+str(len(Z_test)))
            print("Associate to each gaussian a class")
        
        G_train = distribution_function.weighted_gmm_pdf(pi, Z_train, mu, sigma, poincare_function.distance)
        G_train = G_train.max(-1)[1]+1
        
        # for each class we count
        predict_class = torch.zeros(len(mu), len(pi))
        for j, v in enumerate(G_train):
            predict_class[v.item()-1][torch.LongTensor(Y_train[j])-1] +=1 
        sv, si = predict_class.sort(-1)
        g = torch.zeros(len(mu))
        for k in range(len(pi)):
            clas = torch.argmax(predict_class,-1)
            gaus = predict_class[torch.arange(0,len(predict_class)),clas].argmax()
            clas = clas[gaus]
            predict_class[gaus] = -1
            #predict_class[:,clas] = -1
            g[gaus] = clas
        
        # predict
        G_test= distribution_function.weighted_gmm_pdf(pi, Z_test, mu, sigma, poincare_function.distance)
        G_test= G_test.max(-1)[1]+1        

        prediction = g[G_test-1].long()
        acc = accuracy(prediction, torch.LongTensor([i[0]-1 for i in Y_test]))
        acc_total += acc.item()
    return acc_total/(len(I_CV))


def evaluate_em_supervised(Z, Y, n_gaussian, nb_set=5, verbose=False):
    subset_index = torch.randperm(len(Z))
    nb_value = len(Z)//nb_set
    I_CV = [subset_index[nb_value *i:min(nb_value * (i+1), len(Z))] for i in range(nb_set)]
    acc_total = 0.
    for i, test_index in enumerate(I_CV):
        # create train dataset
        train_index = torch.cat([ subset for ci, subset in enumerate(I_CV) if(i!=ci)],0)
        Z_train = Z[train_index]
        Y_train = [Y[ic.item()] for ic in train_index]

        #create test datase
        Z_test = Z[test_index]
        Y_test = [Y[ic.item()] for ic in test_index]        
        
        if(verbose):
            print("Set "+str(i)+" :")
            print("\t train size -> "+str(len(Z_train)))
            print("\t test size -> "+str(len(Z_test)))
            print("Associate to each gaussian a class")
        
        em_alg = RiemannianEM(Z.size(-1), n_gaussian)
        g_mat = torch.Tensor([[ 1 if(y+1 in Y_train[i]) else 0 for y in range(n_gaussian)] for i in range(len(Z_train))])
        em_alg.fit(Z_train, Y=g_mat)
        # predict
        prediction = em_alg.predict(Z_test)

        acc = accuracy(prediction, torch.LongTensor([i[0]-1 for i in Y_test]))
        acc_total += acc.item()
    return acc_total/(len(I_CV))

# in the following function we perform prediction using disc product
# Z, Y, pi, mu, sigma are list of tensor with the size number of disc

def accuracy_supervised(z, y, mu, nb_set=5, verbose=True):
    n_example = len(z)
    n_distrib = len(mu)
    subset_index = torch.randperm(n_example)
    nb_value = n_example//nb_set
    I_CV = [subset_index[nb_value *i:min(nb_value * (i+1), n_example)] for i in range(nb_set)]
    # print(I_CV)
    acc_total = 0.
    for i, test_index in enumerate(I_CV):
        # create train dataset
        train_index = torch.cat([ subset for ci, subset in enumerate(I_CV) if(i!=ci)],0)
        Z_train = z[train_index]
        Y_train = torch.LongTensor([y[ic.item()] for ic in train_index])

        #create test datase
        Z_test = z[test_index]
        Y_test = torch.LongTensor([y[ic.item()] for ic in test_index])      
        
        if(verbose):
            print("Set "+str(i)+" :")
            print("\t train size -> "+str(len(Z_train)))
            print("\t test size -> "+str(len(Z_test)))
            print("Obtaining centroids for each classes")
        
        from function_tools import poincare_alg as pa
        min_label = Y_train.min().item()
        max_label = Y_train.max().item()


        centroids = []
        for i in range(n_distrib):
            # print((Z_train[Y_train[:,0]== (min_label + i)]).size())
            centroids.append(pa.barycenter(Z_train[Y_train[:,0]== (min_label + i)], normed=False).tolist())
        
        centroids = torch.Tensor(centroids).squeeze()
        # predicting 
        Z_test_reshape = Z_test.unsqueeze(1).expand(Z_test.size(0), n_distrib, Z_test.size(-1))
        centroid_reshape = centroids.unsqueeze(0).expand_as(Z_test_reshape)

        d2 = poincare_function.distance(Z_test_reshape, centroid_reshape)**2

        predicted_labels = d2.min(-1)[1] + min_label

        acc = (predicted_labels == Y_test.squeeze()).float().mean()
        acc_total += acc.item()
        print(acc)
    return acc_total/(len(I_CV))


def accuracy_supervised_euclidean(z, y, mu, nb_set=5, verbose=True):
    n_example = len(z)
    n_distrib = len(mu)
    subset_index = torch.randperm(n_example)
    nb_value = n_example//nb_set
    I_CV = [subset_index[nb_value *i:min(nb_value * (i+1), n_example)] for i in range(nb_set)]
    # print(I_CV)
    acc_total = 0.
    for i, test_index in enumerate(I_CV):
        # create train dataset
        train_index = torch.cat([ subset for ci, subset in enumerate(I_CV) if(i!=ci)],0)
        Z_train = z[train_index]
        Y_train = torch.LongTensor([y[ic.item()] for ic in train_index])

        #create test datase
        Z_test = z[test_index]
        Y_test = torch.LongTensor([y[ic.item()] for ic in test_index])      
        
        if(verbose):
            print("Set "+str(i)+" :")
            print("\t train size -> "+str(len(Z_train)))
            print("\t test size -> "+str(len(Z_test)))
            print("Obtaining centroids for each classes")
        
        from function_tools import poincare_alg as pa
        min_label = Y_train.min().item()
        max_label = Y_train.max().item()


        centroids = []
        for i in range(n_distrib):
            print(Z_train[Y_train[:,0]== (min_label + i)].size())
            centroids.append((Z_train[Y_train[:,0]== (min_label + i)].mean(0)).tolist())
        
        centroids = torch.Tensor(centroids).squeeze()
        # predicting 
        Z_test_reshape = Z_test.unsqueeze(1).expand(Z_test.size(0), n_distrib, Z_test.size(-1))
        centroid_reshape = centroids.unsqueeze(0).expand_as(Z_test_reshape)

        d2 = ef.distance(Z_test_reshape, centroid_reshape)**2

        predicted_labels = d2.min(-1)[1] + min_label

        acc = (predicted_labels == Y_test.squeeze()).float().mean()
        acc_total += acc.item()
        print(acc)
    return acc_total/(len(I_CV))

def accuracy_euclidean(z, y, pi, mu, sigma, verbose=False):
    n_disc = len(z)
    n_example = len(z[0])
    n_distrib = len(mu[0])
    y = torch.LongTensor([y[i][0]-1 for i in range(len(y))])
    D = z[0].shape[-1]
    # first getting the pdf for each disc distribution
    def nfunc(sigma):
        return distribution_function.euclidean_norm_factor(sigma, D)
    prob = [distribution_function.weighted_gmm_pdf(pi[i], z[i], mu[i], sigma[i], distance=ef.distance, norm_func=nfunc).unsqueeze(0) 
            for i in range(n_disc)]
    print(torch.cat(prob, 0).shape)
    summed_prob = torch.cat(prob, 0).sum(0)
    print("summed prob size ->",summed_prob.shape)
    _, associated_distrib = summed_prob.max(-1)
    print("associated distribution size ->",associated_distrib.shape)
    print("associated distribution ->",associated_distrib)
    print("source labels ->", y)
    label = associated_distrib.numpy()
    label_source = y.numpy()
    sources_number = n_distrib
    if(n_distrib <= 6):
        return accuracy_small_disc_product(label, label_source, sources_number)
    else:
        return accuracy_huge_disc_product(label, label_source, sources_number)


def accuracy_disc_product(z, y, pi, mu, sigma, verbose=False):
    n_disc = len(z)
    n_example = len(z[0])
    n_distrib = len(mu[0])
    y = torch.LongTensor([y[i][0]-1 for i in range(len(y))])

    # first getting the pdf for each disc distribution
    prob = [distribution_function.weighted_gmm_pdf(pi[i], z[i], mu[i], sigma[i], poincare_function.distance).unsqueeze(0) 
            for i in range(n_disc)]
    print(torch.cat(prob, 0).shape)
    summed_prob = torch.cat(prob, 0).sum(0)
    print("summed prob size ->",summed_prob.shape)
    _, associated_distrib = summed_prob.max(-1)
    print("associated distribution size ->",associated_distrib.shape)
    print("associated distribution ->",associated_distrib)
    print("source labels ->", y)
    label = associated_distrib.numpy()
    label_source = y.numpy()
    sources_number = n_distrib
    if(n_distrib <= 6):
        return accuracy_small_disc_product(label, label_source, sources_number)
    else:
        return accuracy_huge_disc_product(label, label_source, sources_number)

def accuracy_disc_kmeans(z, y, mu, verbose=False):
    n_disc = len(z)
    n_example = len(z)
    n_distrib = len(mu)
    y = torch.LongTensor([y[i][0]-1 for i in range(len(y))])
    from em_tools.kmeans_hyperbolic import PoincareKMeans
    # first getting the pdf for each disc distribution
    kmeans = PoincareKMeans(n_distrib)
    kmeans.fit(z)
    associated_distrib =  kmeans.predict(z.cuda()).cpu()
    # print("associated distribution size ->",associated_distrib.shape)
    # print("associated distribution ->",associated_distrib)
    # print("source labels ->", y)
    label = associated_distrib.numpy()
    label_source = y.numpy()
    sources_number = n_distrib
    std =   kmeans.getStd(z.cuda())
    if(n_distrib <= 6):

        return accuracy_small_disc_product(label, label_source, sources_number), std.max(), std.mean(), std
    else:
        return accuracy_huge_disc_product(label, label_source, sources_number),std.max(), std.mean(), std

def poincare_unsupervised_kmeans(z, y, n_centroid, verbose=False):
    n_example = len(z)
    y = torch.LongTensor([y[i][0]-1 for i in range(len(y))])

    # first getting the pdf for each disc distribution
    kmeans = PoincareKMeans(n_centroid)
    kmeans.fit(z)
    associated_distrib =  kmeans.predict(z)

    label = associated_distrib.numpy()
    label_source = y.numpy()

    std = kmeans.getStd(z)
    if(n_centroid <= 6):
        return accuracy_small_disc_product(label, label_source, n_centroid), std.max(), std.mean(), std
    else:
        return accuracy_huge_disc_product(label, label_source, n_centroid),std.max(), std.mean(), std




def poincare_unsupervised_em(z, y, n_distrib, em=None, verbose=False):
    y = torch.LongTensor([y[i][0]-1 for i in range(len(y))])
    from em_tools.poincare_em import RiemannianEM
    if(em is None):
        em = RiemannianEM( n_distrib, verbose=False)
        em.fit(z, max_iter=5)

    # print(em._mu)
    associated_distrib = em.predict(z)

    label = associated_distrib.numpy()
    label_source = y.numpy()

    if(n_distrib <= 6):
        return accuracy_small_disc_product(label, label_source, n_distrib)
    else:
        return accuracy_huge_disc_product(label, label_source, n_distrib)


def euclidean_unsupervised_em(z, y, n_distrib, em=None, verbose=False):
    y = torch.LongTensor([y[i][0]-1 for i in range(len(y))])
    from em_tools.euclidean_em import GaussianMixtureSKLearn
    print(n_distrib)
    if(em is None):
        em = GaussianMixtureSKLearn( n_distrib)
        em.fit(z)

    # print(em._mu)
    associated_distrib = em.predict(z)

    label = associated_distrib.numpy()
    label_source = y.numpy()

    if(n_distrib <= 6):
        return accuracy_small_disc_product(label, label_source, n_distrib)
    else:
        return accuracy_huge_disc_product(label, label_source, n_distrib)
        
def accuracy_euclidean_kmeans(z, y, mu, verbose=False):
    n_disc = len(z)
    n_example = len(z)
    n_distrib = len(mu)
    y = torch.LongTensor([y[i][0]-1 for i in range(len(y))])
    from sklearn.cluster import KMeans
    # first getting the pdf for each disc distribution
    kmeans = KMeans(n_distrib, n_init=1, init="random")
    kmeans.fit(z.numpy())
    associated_distrib =  kmeans.predict(z.numpy())
    # print("associated distribution size ->",associated_distrib.shape)
    # print("associated distribution ->",associated_distrib)
    # print("source labels ->", y)
    centroids = torch.Tensor(kmeans.cluster_centers_)
    N, K, D = z.shape[0], centroids.shape[0], z.shape[1]
    centroids = centroids.unsqueeze(0).expand(N, K, D)
    x = z.unsqueeze(1).expand(N, K, D)
    dst =(centroids-x).norm(2,-1)**2
    value, indexes = dst.min(-1)
    stds = []
    for i in range(n_distrib):
        stds.append(value[indexes==i].sum())
    std  = torch.Tensor(stds)
    label = associated_distrib
    label_source = y.numpy()
    sources_number = n_distrib
    if(n_distrib <= 6):
        return accuracy_small_disc_product(label, label_source, sources_number), std.tolist()
    else:
        return accuracy_huge_disc_product(label, label_source, sources_number), std.tolist()
def accuracy_small_disc_product(label, label_source, sources_number):
    combinations = []
    zero_fill_comb = np.zeros(len(label))

    Numbers =  np.arange(0, sources_number)
    numerotations = list(itertools.permutations(Numbers))

    # print("zeroçfcom", len(label))

    for i in range(0,math.factorial(sources_number)):
        combinations.append(zero_fill_comb)


    combinations = np.array(combinations)
    numerotations = np.array(numerotations)


    for i in range(0,len(combinations)):
         combinations[i] = label_source.copy()



    # Calcul des tableaux permutés
    for i in range (0,len(numerotations)):

        # print('i',i)
        # print('numerotation\n', numerotations[i])
        for j in range(0,len(combinations[i])):

            for q in range(0,len(Numbers)):
                if(combinations[i][j]== Numbers[q]):
                    combinations[i][j] = numerotations[i][q]
                    break



    # print('Combinations after permutations\n',combinations)

    result = np.zeros(len(combinations[0]))

    # print('Len result',len(combinations[:,0]))


    result_percentage = []

    for u in range(0,len(combinations[:,0])):

        result_combination = (combinations[u]-label)

        # print('result combination', result_combination)

        np.append(result, result_combination)



        result_int = (sum(1 for i in result_combination if i == 0) / len(label_source)) * 100

        # print('sum(1 for i in result_combination if i == 0)',sum(1 for i in result_combination if i == 0))

        result_percentage.append(result_int)



    # print('result',result_percentage)
    return max(result_percentage)

def accuracy_huge_disc_product(label, label_source, sources_number):

    numerotation_initial = np.zeros(sources_number, dtype=int)

    numerotation_initial = numerotation_initial - 1

    # print('Numerotation initial\n', numerotation_initial)

    number_data_per_cluster = np.zeros(sources_number, dtype=int)

    priority_clusters = np.zeros(sources_number, dtype=int)

    for j in range(0, len(priority_clusters)):
        priority_clusters[j] = j

    # print('Priority Cluster\n', priority_clusters)

    # Pour chaque cluster calculé
    for i in range(0, sources_number):
        for j in range(0, len(label)):
            if (label[j] == i):
                # On calcul le nombre de données par Cluster
                number_data_per_cluster[i] = number_data_per_cluster[i] + 1

                # Pour chaque donnée qui appartient à ce Cluster
                # On va voir le cluster de la verite de terrain et compter

    # print('Number Data per cluster\n',number_data_per_cluster)

    # On va classer les clusters selon le nombre de donnees qu'ils contiennent
    # Par ordre decroissant

    for q in range(0, len(priority_clusters)):
        for u in range(q + 1, len(priority_clusters)):
            if (number_data_per_cluster[priority_clusters[q]] < number_data_per_cluster[priority_clusters[u]]):
                temp = priority_clusters[q].copy()
                priority_clusters[q] = priority_clusters[u].copy()
                priority_clusters[u] = temp.copy()

    # print('Priority Clusters after\n',priority_clusters)

    # On commence par le cluster le plus prioritaire A (plus de donnnes)

    taken_or_not = []
    for i in range(0, sources_number):
        taken_or_not.append(False)

    for i in range(0, len(priority_clusters)):

        # On cherche le noeud de la verite de terrain qui apparait le plus de fois dans A
        count = np.zeros(sources_number, dtype=int)

        for j in range(0, len(label)):

            # Pour chaque donnée qui appartient à A
            if (label[j] == priority_clusters[i]):
                count[label_source[j]] = count[label_source[j]] + 1

        # print('Count for cluster',priority_clusters[i],'is\n',count)

        max_count = 0
        for q in range(0, len(count)):
            if (count[q] >= max_count and taken_or_not[q] == False):
                max_count = count[q]
                numerotation_initial[priority_clusters[i]] = q

        taken_or_not[numerotation_initial[priority_clusters[i]]] = True

    # print('Large K Guess\n', numerotation_initial)

    # Maintenant faut remplacer les numéros de label par ceux du numerotation_initial

    new_label = label.copy()

    for j in range(0, len(new_label)):
        for q in range(0, len(numerotation_initial)):
            if (new_label[j] == q):
                new_label[j] = numerotation_initial[q]
                break

    # print('New label\n')
    # for i in new_label:
    #    print(i)

    # print('Source label\n')
    # for i in label_source:
    #   print(i)

    result_combination = (new_label - label_source)

    max_result = (sum(1 for i in result_combination if i == 0) / len(label_source)) * 100

    # print('Max result large K',max_result)

    # Apres il faut faire les permutations sur les classes de tout ceux qui sont pas taken

    # untaken = []
    #
    # print('taken or not\n',taken_or_not)
    #
    # for i in range(0,len(taken_or_not)):
    #     if taken_or_not[i] == False:
    #         print(i)
    #         untaken.append(i)
    #
    # print('untaken\n',untaken)
    #
    # numerotations = list(itertools.permutations(untaken))
    #
    # print('Numerotations', numerotations)

    return max_result

########################################### TEST ##############################################

def test_cross_val():
    from data_tools import corpora
    import os
    filepath =  "/local/gerald/POINCARE-EM/DT/dblp-2D-EM-TEST-13"
    dataset_dict = { "karate": corpora.load_karate,
            "football": corpora.load_football,
            "flickr": corpora.load_flickr,
            "dblp": corpora.load_dblp,
            "books": corpora.load_books,
            "blogCatalog": corpora.load_blogCatalog,
            "polblog": corpora.load_polblogs,
            "adjnoun": corpora.load_adjnoun
          }
    D, X, Y = dataset_dict["dblp"]()
    # transform labels tor torch zeros-ones tensor
    ground_truth = torch.LongTensor([[ 1 if(y+1 in Y[i]) else 0 for y in range(5)] for i in range(len(X))])
    print(ground_truth[:,0].sum())
    embeddings   = torch.load(os.path.join(filepath,"embeddings.t7"))[0]
    scoring_function = PrecisionScore(at=1)
    CVE = CrossValEvaluation(embeddings, ground_truth, nb_set=2, algs_object=RiemannianEM)
    scores = CVE.get_score(scoring_function)
    print("scores -> ", scores)
    print("mean score -> ", sum(scores,0)/2)

def test_mean_conductance():
    from data_tools import corpora
    import os
    print("------------------------MConductance------------------------")
    print("Testing on a fake examples (Testing execution)")
    graph = {0:[4,5,7], 
            1:[2,5,3], 
            2:[7,5,7], 
            3:[4,5,0],
            4:[1,5,7,2],
            5:[4,5,6,7], 
            6:[0,1,7],
            7:[6,3,1]}
    prediction = torch.rand(8, 5)
    prediction[torch.arange(8),prediction.max(-1)[1]] = 1
    prediction[prediction<1] = 0

    score = mean_conductance(prediction, graph)

    print("score (conductance on fake data) -> "+str(score))

    filepath =  "/local/gerald/POINCARE-EM/DT/dblp-2D-EM-TEST-13"
    dataset_dict = { "karate": corpora.load_karate,
            "football": corpora.load_football,
            "flickr": corpora.load_flickr,
            "dblp": corpora.load_dblp,
            "books": corpora.load_books,
            "blogCatalog": corpora.load_blogCatalog,
            "polblog": corpora.load_polblogs,
            "adjnoun": corpora.load_adjnoun
          }
    D, X, Y = dataset_dict["dblp"]()
    # transform labels tor torch zeros-ones tensor
    # ground_truth = torch.LongTensor([[ 1 if(y+1 in Y[i]) else 0 for y in range(5)] for i in range(len(X))])
    # print(ground_truth[:,0].sum())
    embeddings  = torch.load(os.path.join(filepath,"embeddings.t7"))[0]
    algs = RiemannianEM(5)
    algs.fit(embeddings)
    prediction = algs.predict(embeddings)
    print(prediction)

    prediction_mat = torch.LongTensor([[ 1 if(y == prediction[i]) else 0 for y in range(5)] for i in range(len(X))])
    adjency_matrix = X
    # print(prediction_mat.sum(0))
    score = mean_conductance(prediction_mat, adjency_matrix)
    print("scores -> ", score)

def test_nmi():
    from data_tools import corpora
    import os
    print("------------------------NMI------------------------")
    print("Testing on a fake examples (Testing execution)")
    graph = {0:[4,5,7], 
            1:[2,5,3], 
            2:[7,5,7], 
            3:[4,5,0],
            4:[1,5,7,2],
            5:[4,5,6,7], 
            6:[0,1,7],
            7:[6,3,1]}
    ground_truth = torch.LongTensor([[ 1 if(y in graph[i]) else 0 for y in range(5)] for i in range(len(graph))])
    prediction = torch.rand(8, 5)
    prediction[torch.arange(8),prediction.max(-1)[1]] = 1
    prediction[prediction<1] = 0

    score = nmi(prediction, ground_truth)

    print("score (nmi on fake data) -> "+str(score))

    filepath =  "/local/gerald/POINCARE-EM/DT/dblp-2D-EM-TEST-13"
    dataset_dict = { "karate": corpora.load_karate,
            "football": corpora.load_football,
            "flickr": corpora.load_flickr,
            "dblp": corpora.load_dblp,
            "books": corpora.load_books,
            "blogCatalog": corpora.load_blogCatalog,
            "polblog": corpora.load_polblogs,
            "adjnoun": corpora.load_adjnoun
          }
    D, X, Y = dataset_dict["dblp"]()
    # transform labels tor torch zeros-ones tensor
    ground_truth = torch.LongTensor([[ 1 if(y+1 in Y[i]) else 0 for y in range(5)] for i in range(len(X))])
    # print(ground_truth[:,0].sum())
    embeddings  = torch.load(os.path.join(filepath,"embeddings.t7"))[0]
    algs = RiemannianEM(5)
    algs.fit(embeddings)
    prediction = algs.predict(embeddings)
    prediction_mat = torch.LongTensor([[ 1 if(y == prediction[i]) else 0 for y in range(5)] for i in range(len(X))])
    adjency_matrix = X
    # print(prediction_mat.sum(0))
    score = nmi(prediction_mat, ground_truth)
    print("scores -> ", score)   

###############################################

if __name__ == "__main__":
    # execute only if run as a script
    # test_cross_val()
    # test_mean_conductance()
    test_nmi()