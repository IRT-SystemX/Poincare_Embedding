
import itertools
import math
import numpy as np


def Truth_Check_Small_K(filename,label, sources_number = 2):

    file = open('Input/R_'+filename)

    #print('Labels\n',label)
    label_source = []
    for line in file.readlines():
        label_source.append(line)

    label_source = np.array(label_source)
    label_source = label_source.astype(int)

    label_source = label_source-1

    #print('Label source',label_source)
    #print('Label obtenu', label)

    combinations = []
    zero_fill_comb = np.zeros(len(label))

    Numbers =  np.arange(0,sources_number)
    numerotations = list(itertools.permutations(Numbers))



    for i in range(0,math.factorial(sources_number)):
        combinations.append(zero_fill_comb)


    combinations = np.array(combinations)
    numerotations = np.array(numerotations)


    for i in range(0,len(combinations)):
         combinations[i] = label_source.copy()



    #Calcul des tableaux permutés
    for i in range (0,len(numerotations)):

        #print('i',i)
        #print('numerotation\n', numerotations[i])
        for j in range(0,len(combinations[i])):

            for q in range(0,len(Numbers)):
                if(combinations[i][j]== Numbers[q]):
                    combinations[i][j] = numerotations[i][q]
                    break



    #print('Combinations after permutations\n',combinations)

    result = np.zeros(len(combinations[0]))

    #print('Len result',len(combinations[:,0]))


    result_percentage = []

    for u in range(0,len(combinations[:,0])):

        result_combination = (combinations[u]-label)

        #print('result combination', result_combination)

        np.append(result, result_combination)



        result_int = (sum(1 for i in result_combination if i == 0) / len(label_source)) * 100

        #print('sum(1 for i in result_combination if i == 0)',sum(1 for i in result_combination if i == 0))

        result_percentage.append(result_int)



   # print('result',result_percentage)


    max_result = max(result_percentage)

    print('Max result',max_result)

    return max_result