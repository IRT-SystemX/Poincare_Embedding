CUDA_VISIBLE_DEVICES=2 python3.7 launcher_tools/experiment_poincare.py  --distance-coef 0.5 --dataset dblp  --n-gaussian 12 --walk-lenght 20 --precompute-rw 1  --epoch 50 --epoch-embedding 10  --epoch-embedding-init 20 --beta 1.  --alpha 1. --gamma .1 --lr 3e-3   --negative-sampling  10  --context-size 3   --cuda --embedding-optimizer exphsgd  --em-iter 10  --id dblp-2D-EM-05 --size 2 --seed 220
CUDA_VISIBLE_DEVICES=2 python3.7 launcher_tools/experiment_poincare.py  --distance-coef 0.7 --dataset dblp  --n-gaussian 12 --walk-lenght 20 --precompute-rw 1  --epoch 50 --epoch-embedding 10  --epoch-embedding-init 20 --beta 1.  --alpha 1. --gamma .1 --lr 3e-3   --negative-sampling  10  --context-size 3   --cuda --embedding-optimizer exphsgd  --em-iter 10  --id dblp-2D-EM-06 --size 2 --seed 220
CUDA_VISIBLE_DEVICES=2 python3.7 launcher_tools/experiment_poincare.py  --distance-coef 1.0 --dataset dblp  --n-gaussian 12 --walk-lenght 20 --precompute-rw 1  --epoch 50 --epoch-embedding 10  --epoch-embedding-init 20 --beta 1.  --alpha 1. --gamma .1 --lr 3e-3   --negative-sampling  10  --context-size 3   --cuda --embedding-optimizer exphsgd  --em-iter 10  --id dblp-2D-EM-07 --size 2 --seed 220
