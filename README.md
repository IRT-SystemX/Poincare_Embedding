# Poincaré Expectation Maximisation 

This package implements the expectation maximisation algorithm for the Poincaré manifold. 
A multi-dimensional setting is also implemented using the products of many Poincaré disks. 


![alt text](https://github.com/hz789/EM_Hyperbolic/blob/master/Readme_Figure.png?raw=true "Gaussien mixture model")
Poincaré disk with two Gaussians computed via graph embedding followed by exptectation maximisation algorithm for the Karate example dataset.


![alt text](https://github.com/tgeral68/EM_Hyperbolic/tree/master/figures/football_example_m.pdf?raw=true "Gaussien mixture model on Football")
Poincaré disks with 12 Gaussians computed via graph embedding followed by exptectation maximisation algorithm for the Football example dataset.

### The package contains the following modules:

* EM_Algorithm: implements the expectation maximisation algorithm
* EM_Embedding: implements the embedding of a graph structured data given an adjacency matrix then applies the EM algorithm 



### Package operation

Insert the dataset graph example as an adjacency matrix in the Input folder.
Open the Run_Embedding+EM.py script and fill the parameters.
Launch the script.
Check results in the Output folder. 


## Dependencies

> pytorch sklearn tqdm 

## Exemple
launching an experiment with several discs
> python launcher_tools/experiment.py --n-disc 5 --epoch-embedding 200 --epoch 1 --dataset football --n-gaussian 12 --beta 1.0 --lr 10

<details><summary>Click here for the rest of available datasets</summary>
<p>

> python launcher_tools/experiment.py --n-disc 5 --epoch-embedding 200 --epoch 1 --dataset football --n-gaussian 12 --beta 1.0 --lr 10 <br/>
> python launcher_tools/experiment.py --n-disc 5 --epoch-embedding 200 --epoch 1 --dataset karate --n-gaussian 2 --beta 1.0 --lr 10 <br/>
> python launcher_tools/experiment.py --n-disc 5 --epoch-embedding 200 --epoch 1 --dataset flickr --n-gaussian ? --beta 1.0 --lr 10 <br/>
> python launcher_tools/experiment.py --n-disc 5 --epoch-embedding 200 --epoch 1 --dataset dblp --n-gaussian 5 --beta 1.0 --lr 10 <br/>
> python launcher_tools/experiment.py --n-disc 5 --epoch-embedding 200 --epoch 1 --dataset books --n-gaussian 3 --beta 1.0 --lr 10 <br/>
> python launcher_tools/experiment.py --n-disc 5 --epoch-embedding 200 --epoch 1 --dataset blogCatalog --n-gaussian 2 --beta 1.0 --lr 10 <br/>

</p>
</details>

## Other dataset
Currently it does not work on large scale dataset, you can download it in just calling the dataset function in /data_tools/corpora.py (end of the file). Or you can directly launch the main script with "parameters.dataset_loading_func = corpora.load_flickr" (for flickr dataset)
