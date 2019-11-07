'''
A script to transform dataset in mat format (for ComE code)
'''
import argparse
from os import path
from data_tools import corpora_tools, corpora, data, logger


parser = argparse.ArgumentParser(description='Load embeddings and perform kmeans on it')

parser.add_argument('--folder', dest="folder", type=str, default="/local/gerald/data/",
                    help="embeddings location file") 
parser.add_argument('--dataset', dest="dataset", type=str, default="dblp",
                    help="dataset") 

args = parser.parse_args()


dataset_dict = { "karate": corpora.load_karate,
            "football": corpora.load_football,
            "flickr": corpora.load_flickr,
            "dblp": corpora.load_dblp,
            "books": corpora.load_books,
            "blogCatalog": corpora.load_blogCatalog,
            "polblog": corpora.load_polblogs,
            "adjnoun": corpora.load_adjnoun
          }
print("Loading the dataset")
D, X, Y = dataset_dict[args.dataset]()
print("Saving the dataset")
corpora.saving_matlab_corpus(X, Y, 
                             path.join(args.folder, args.dataset+".mat" ),
                             path.join(args.folder, args.dataset+".labels" )
                            )

