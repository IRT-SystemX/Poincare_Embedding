import torch
from function_tools import gmm_tools, modules
from collections import Counter

def accuracy(prediction, labels):
    return (prediction == labels).float().mean()

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
        
        G_train = gmm_tools.weighted_gmm_pdf(pi, Z_train, mu, sigma, modules.hyperbolicDistance)
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
        G_test= gmm_tools.weighted_gmm_pdf(pi, Z_test, mu, sigma, modules.hyperbolicDistance)
        G_test= G_test.max(-1)[1]+1        

        prediction = g[G_test-1].long()
        acc = accuracy(prediction, torch.LongTensor([i[0]-1 for i in Y_test]))
        acc_total += acc.item()
    return acc_total/(len(I_CV))