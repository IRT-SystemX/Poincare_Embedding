CUDA_VISIBLE_DEVICES=0 python3.7 launcher_tools/experiment_poincare_alternate.py --loss-aggregation sum --distance-coef 1. --dataset blogCatalog  --n-gaussian 39 --walk-lenght 60 --precompute-rw 4  --epoch 1 --epoch-embedding 1  --epoch-embedding-init 10 --beta 5e-4  --alpha 1e-4 --gamma 1e-4 --lr  1. --negative-sampling 5  --context-size 5  --cuda --embedding-optimizer exphsgd  --em-iter 1 --id blogCatalog-10D-1  --size 10  --seed 246 --batch-size 512

python3.7 launcher_tools/evaluation_classifier_poincare.py --file /local/gerald/POINCARE-EM/blogCatalog-10D-1

CUDA_VISIBLE_DEVICES=0 python3.7 launcher_tools/experiment_poincare_alternate.py --loss-aggregation sum --distance-coef 1. --dataset blogCatalog  --n-gaussian 39 --walk-lenght 60 --precompute-rw 4  --epoch 1 --epoch-embedding 1  --epoch-embedding-init 10 --beta 5e-4  --alpha 1e-4 --gamma 1e-4 --lr  1. --negative-sampling 5  --context-size 5  --cuda --embedding-optimizer exphsgd  --em-iter 1 --id blogCatalog-10D-2  --size 10  --seed 247 --batch-size 512

python3.7 launcher_tools/evaluation_classifier_poincare.py --file /local/gerald/POINCARE-EM/blogCatalog-10D-2

CUDA_VISIBLE_DEVICES=0 python3.7 launcher_tools/experiment_poincare_alternate.py --loss-aggregation sum --distance-coef 1. --dataset blogCatalog  --n-gaussian 39 --walk-lenght 60 --precompute-rw 4  --epoch 1 --epoch-embedding 1  --epoch-embedding-init 10 --beta 5e-4  --alpha 1e-4 --gamma 1e-4 --lr  1. --negative-sampling 5  --context-size 5  --cuda --embedding-optimizer exphsgd  --em-iter 1 --id blogCatalog-10D-3  --size 10  --seed 248 --batch-size 512

python3.7 launcher_tools/evaluation_classifier_poincare.py --file /local/gerald/POINCARE-EM/blogCatalog-10D-3

CUDA_VISIBLE_DEVICES=0 python3.7 launcher_tools/experiment_poincare_alternate.py --loss-aggregation sum --distance-coef 1. --dataset blogCatalog  --n-gaussian 39 --walk-lenght 60 --precompute-rw 4  --epoch 1 --epoch-embedding 1  --epoch-embedding-init 10 --beta 5e-4  --alpha 1e-4 --gamma 1e-4 --lr  1. --negative-sampling 5  --context-size 5  --cuda --embedding-optimizer exphsgd  --em-iter 1 --id blogCatalog-10D-4  --size 10  --seed 249 --batch-size 512

python3.7 launcher_tools/evaluation_classifier_poincare.py --file /local/gerald/POINCARE-EM/blogCatalog-10D-4

CUDA_VISIBLE_DEVICES=0 python3.7 launcher_tools/experiment_poincare_alternate.py --loss-aggregation sum --distance-coef 1. --dataset blogCatalog  --n-gaussian 39 --walk-lenght 60 --precompute-rw 4  --epoch 1 --epoch-embedding 1  --epoch-embedding-init 10 --beta 5e-4  --alpha 1e-4 --gamma 1e-4 --lr  1. --negative-sampling 5  --context-size 5  --cuda --embedding-optimizer exphsgd  --em-iter 1 --id blogCatalog-10D-5  --size 10  --seed 250 --batch-size 512

python3.7 launcher_tools/evaluation_classifier_poincare.py --file /local/gerald/POINCARE-EM/blogCatalog-10D-5



