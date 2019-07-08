
from EM_Algorithm.EM import *
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




    output_directory = 'Output/' +file_name+ '/' + file_name + 'Hyperbolic_multidim_process_number_'+str(process_number) + '/Test_nstep_' + str(Embedding_Parameters['nstep']) + '_nepoch_' + str(
        Embedding_Parameters['nepoch']) \
                       + '_context_' + str(Embedding_Parameters['context']) + '_negsample_' \
                       + str(Embedding_Parameters['negsample']) + '_p_' + str(Embedding_Parameters['p_gradient']) +  str(Embedding_Parameters['number_poincare_disks']) + '/'

    try:
        os.makedirs(output_directory)
        print("Directory ", output_directory, " Created")

    except FileExistsError:
        print("Directory ", output_directory, " already exists")

    print('Saving data...')
    with open(output_directory + 'Output_Data' + file_name + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([A, Embedding_table, R], f)


    if(Plot_Parameters['plot_or_no']):

        for y in range(len(Embedding_table)):

            # Plot of the embedding

            Plot_Embedding_Poincare_Multidim(Embedding_table[y], output_directory, y, False)



    # EM( EM_Parameters['iter_max'],
    #     EM_Parameters['M'],
    #     file_name,
    #     'Input/',
    #     Performance_Computation_Parameters['truth_check']
    #     )

    return True


