CUDA_VISIBLE_DEVICES=0 python3.5 launcher_Windows.py --loss-aggregation mean --distance-coef 1. --dataset wikipedia  --n-gaussian 40 --walk-lenght 80 --precompute-rw 5  --epoch 10 --epoch-embedding 50  --epoch-embedding-init 100 --beta 1.  --alpha 1. --gamma 1. --lr 1  --negative-sampling 5  --context-size 10 --embedding-optimizer exphsgd  --em-iter 20  --id wikipedia-2D-EM-TEST-19  --size 2  --seed 246 --batch-size 40000
