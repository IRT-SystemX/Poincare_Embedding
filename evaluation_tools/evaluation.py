import torch
from function_tools import distribution_function, poincare_function
from collections import Counter
import numpy as np
import math
import itertools

def accuracy(prediction, labels):
    return (prediction == labels).float().mean()

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

# in the following function we perform prediction using disc product
# Z, Y, pi, mu, sigma are list of tensor with the size number of disc

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



    print('result',result_percentage)
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
