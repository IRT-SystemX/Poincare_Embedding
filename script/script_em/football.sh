CUDA_VISIBLE_DEVICES=0 python3.7 launcher_tools/experiment_poincare_alternate.py  --distance-coef 0.5 --dataset football  --n-gaussian 12 --walk-lenght 10 --precompute-rw 10 --epoch 10 --epoch-embedding 100  --epoch-embedding-init 500 --beta .5  --alpha 5. --gamma 2 --lr 1e-3   --negative-sampling 10  --context-size 4  --cuda --embedding-optimizer exphsgd  --em-iter 10  --id football-2D-EMTEST-07 --size 2 --seed 232 --batch-size 40000
CUDA_VISIBLE_DEVICES=0 python3.7 launcher_tools/experiment_poincare_alternate.py  --distance-coef 0.5 --dataset football  --n-gaussian 12 --walk-lenght 10 --precompute-rw 10 --epoch 10 --epoch-embedding 100  --epoch-embedding-init 500 --beta .5  --alpha 5. --gamma 2 --lr 1e-3   --negative-sampling 10  --context-size 5 --cuda --embedding-optimizer exphsgd  --em-iter 10  --id football-2D-EMTEST-08 --size 2 --seed 233 --batch-size 40000