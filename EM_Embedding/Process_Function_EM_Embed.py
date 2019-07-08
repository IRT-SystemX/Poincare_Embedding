
from EM_Algorithm.EM import *

def EM_Embedding_Process (example_name,
                            Embedding_Parameter,
                            EM_Parameters,
                            Performance_Computation_Parameters,
                            Plot_Parameters,
                            process_number):





    EM( EM_Parameters['iter_max'],
        EM_Parameters['M'],
        example_name,
        filename,
        Performance_Computation_Parameters['truth_check']
        )
    return True


