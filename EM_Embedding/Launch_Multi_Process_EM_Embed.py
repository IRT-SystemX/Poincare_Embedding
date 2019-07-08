#The following functions launches multiple processes to perform different embedding seperately

from multiprocessing import Process, Manager
from EM_Embedding.Process_Function_EM_Embed import *


def Launch_multiprocess_EM_Embed (number_processes, file_name, Embedding_Parameters, EM_Parameters, Performance_Computation_Parameters, Plot_Parameters):

    ######### This part #############################
    #### Launches different processes for hyperbolic embedding #######
    ##################################################################

    manager = Manager()

    processes = []

    manager.list(range(number_processes))

    #print('Initial Random_walks\n', Random_Walks)

    for i in range(number_processes):

        p = Process(target = EM_Embedding_Process,
                    args = (file_name,
                            Embedding_Parameters,
                            EM_Parameters,
                            Performance_Computation_Parameters,
                            Plot_Parameters,
                            i
                            ))
        p.start()
        processes.append(p)

    for p in processes:

        p.join()

    return True