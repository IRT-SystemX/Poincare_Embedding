# Poincaré Expectation Maximisation 

This package implements the expectation maximisation algorithm for the Poincaré manifold. 
A multi-dimensional setting is also implemented using the products of many Poincaré disks. 


![alt text](https://github.com/hz789/EM_Hyperbolic/blob/master/Readme_Figure.png?raw=true "Gaussien mixture model")
Poincaré disk with two Gaussians computed via graph embedding followed by exptectation maximisation algorithm for the Karate example dataset.


![alt text](https://github.com/tgeral68/EM_Hyperbolic/blob/master/figures/football_example_m.png?raw=True)
Poincaré disks with 12 Gaussians computed via graph embedding followed by exptectation maximisation algorithm for the Football example dataset.

### The package contains the following modules:

* EM_Algorithm: implements the expectation maximisation algorithm
* EM_Embedding: implements the embedding of a graph structured data given an adjacency matrix then applies the EM algorithm 
https://github.com/tgeral68/EM_Hyperbolic/tree/master/figures/football_example_m.png?raw=true


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
> python launcher_tools/experiment.py --n-disc 5 --epoch-embedding 200 --epoch 1 --dataset adjnoun --n-gaussian 2 --beta 1.0 --lr 10 <br/>
> python launcher_tools/experiment.py --n-disc 5 --epoch-embedding 200 --epoch 1 --dataset polblogs --n-gaussian 2 --beta 1.0 --lr 10 <br/>

</p>
</details>

## Other dataset
Currently it does not work on large scale dataset, you can download it in just calling the dataset function in /data_tools/corpora.py (end of the file). Or you can directly launch the main script with "parameters.dataset_loading_func = corpora.load_flickr" (for flickr dataset)


### MIT License

Copyright 2019 Hatem Hajri, Hadi Zaatiti, Thomas Gérald, Georges Hébrail

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


