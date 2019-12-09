CUDA_VISIBLE_DEVICES=0 python3.7 launcher_tools/experiment_poincare_alternate.py --loss-aggregation sum --distance-coef 1. --dataset blogCatalog  --n-gaussian 39 --walk-lenght 60 --precompute-rw 4  --epoch 1 --epoch-embedding 1  --epoch-embedding-init 10 --beta .001  --alpha .01 --gamma 1. --lr  1. --negative-sampling 5  --context-size 5  --cuda --embedding-optimizer exphsgd  --em-iter 1 --id blogCatalog-2D-EM-TEST-1  --size 2  --seed 246 --batch-size 512

# python3.7 launcher_tools/evaluation_supervised_poincare.py --file /local/gerald/POINCARE-EM/blogCatalog-2D-EM-TEST-2

python3.7 launcher_tools/evaluation_classifier_poincare.py --file /local/gerald/POINCARE-EM/blogCatalog-2D-EM-TEST-1