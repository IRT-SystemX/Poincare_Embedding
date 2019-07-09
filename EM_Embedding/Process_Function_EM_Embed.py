from EM_Algorithm.EM_from_embedding import *
from EM_Embedding.Embedding_Multidim import *
from EM_Algorithm.Read_From_File import *
from Visualization.Gaussian_Mixture_Visualization import *
from Visualization.Gaussian_Visualization import *
from Visualization.Plot_Embedding_Poincare import *


def EM_Embedding_Process (file_name,
                            Embedding_Parameters,
                            EM_Parameters,
                            Performance_Computation_Parameters,
                            Plot_Parameters,
                            process_number):

    f = open('Input/' + file_name)
    A = []
    for line in f.readlines():
        line2 = line.split()
        A.append(line2)

    A = np.array(A)
    A = A.astype(int)

    n = len(A)
    print('Data matrix A contains: ', n, ' nodes.')

    print('Embedding A in ', Embedding_Parameters['number_poincare_disks'], ' Poincaré disks...')

    Embedding_table, R = Embedding_Multidim_function(A, Embedding_Parameters['nstep'], Embedding_Parameters['nepoch'],
                                                     Embedding_Parameters['context'], Embedding_Parameters['p_gradient'],
                                                     Embedding_Parameters['negsample'], Embedding_Parameters['number_poincare_disks'])

    Z = []

    for g in range(len(Embedding_table)):
            Z.append(1j * Embedding_table[g][:, 1])
            Z[g] += Embedding_table[g][:, 0]


    weights_table, variances_table, barycentres_table = EM( EM_Parameters['iter_max'],
        EM_Parameters['M'],
        Embedding_table
        )

    #Find the class for each data node by applying the criterion

    labels = np.empty(n)

    for i in range(n):

        probabilities = np.zeros(EM_Parameters['M'])

        for j in range(EM_Parameters['M']):

            for dimension_index in range(len(Z)):

                probabilities[j] = probabilities[j] +  weights_table[dimension_index][j] * Gaussian_PDF(Z[i], barycentres_table[dimension_index][j], variances_table[dimension_index][j])

        labels[i] = np.argmax(probabilities)

    print('labels',labels)

    #Ground Truth Check
    if(Performance_Computation_Parameters['truth_check'] == True):

        if(EM_Parameters['M']>6):
            performance = Truth_Check_Large_K(file_name,labels,EM_Parameters['M'])
        else:

            performance = Truth_Check_Small_K(file_name,labels, EM_Parameters['M'])


        print('Performance', performance)




    #Save everything and plot

    #Creation of the output directory if it does not yet exists

    output_directory = 'Output/'+example_name+'/'+filename+'_'+str(iter)+'_iter_'+str(M)+'_M'
    try:
        os.makedirs(output_directory)
        print("Directory ", output_directory, " Created")

    except FileExistsError:
        print("Directory ", output_directory, " already exists")


    #Save to Python Pickle
    print('Saving data...')
    with open(output_directory + '/Output_Data_' + example_name + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([B, weights, barycentres, variances, performance], f)

    print('Data Saved to ', output_directory + '/Output_Data_' +example_name+ '.pkl')

    #Save to text file

    print('Saving Variances and Truth percentage to text file...')
    file = open(output_directory + '/Performances_'+example_name +'.txt', 'w')


    # file.write('Variances:\n')
    # for i in Variances:
    #     file.write(str(i) + '\n')
    file.write('Truth percentage:\n')
    file.write(str(performance))

    file.close()



    #Plotting the results

    Plot_Gaussian_Mixture(Z, barycentres, variances, weights, output_directory)




    output_directory = 'Output/' +file_name+ '/' + file_name + 'Hyperbolic_EM_Embed_process_number_'+str(process_number) + '/Test_nstep_' + str(Embedding_Parameters['nstep']) + '_nepoch_' + str(
        Embedding_Parameters['nepoch']) \
                       + '_context_' + str(Embedding_Parameters['context']) + '_negsample_' \
                       + str(Embedding_Parameters['negsample']) + '_p_' + str(Embedding_Parameters['p_gradient']) +'_Disk_Number_'  + str(Embedding_Parameters['number_poincare_disks']) + '_Class_Numb_'
    + str(EM_Parameters['M']) + '_EM_Iter_'  + str(EM_Parameters['M']) + '/'

    try:
        os.makedirs(output_directory)
        print("Directory ", output_directory, " Created")

    except FileExistsError:
        print("Directory ", output_directory, " already exists")

    print('Saving data...')
    with open(output_directory + 'Output_Data' + file_name + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([A, Embedding_table, R], f)


    if(Plot_Parameters['plot_or_no']):
        print('Plotting...')
        for y in range(len(Embedding_table)):

            # Plot of the embedding

            Plot_Embedding_Poincare_Multidim(Embedding_table[y], output_directory, y, False)

        print('Done Plotting!')



    return True


