CUDA_VISIBLE_DEVICES=2 python3.7 launcher_tools/experiment_poincare_alternate.py --loss-aggregation sum --distance-coef 1. --dataset wikipedia --n-gaussian 40 --walk-lenght 60 --precompute-rw 4  --epoch 5 --epoch-embedding 1  --epoch-embedding-init 20 --beta .001  --alpha .0005 --gamma .001 --lr  1. --negative-sampling 5  --context-size 5  --cuda --embedding-optimizer exphsgd  --em-iter 5 --id Wikipedia-5D-1  --size 5  --seed 246 --batch-size 512

python3.7 launcher_tools/evaluation_classifier_poincare.py --file /local/gerald/POINCARE-EM/Wikipedia-5D-1

CUDA_VISIBLE_DEVICES=2 python3.7 launcher_tools/experiment_poincare_alternate.py --loss-aggregation sum --distance-coef 1. --dataset wikipedia --n-gaussian 40 --walk-lenght 60 --precompute-rw 4  --epoch 5 --epoch-embedding 1  --epoch-embedding-init 20 --beta .001  --alpha .0005 --gamma .001 --lr  1. --negative-sampling 5  --context-size 5  --cuda --embedding-optimizer exphsgd  --em-iter 5 --id Wikipedia-5D-2  --size 5  --seed 247 --batch-size 512

python3.7 launcher_tools/evaluation_classifier_poincare.py --file /local/gerald/POINCARE-EM/Wikipedia-5D-2

CUDA_VISIBLE_DEVICES=2 python3.7 launcher_tools/experiment_poincare_alternate.py --loss-aggregation sum --distance-coef 1. --dataset wikipedia --n-gaussian 40 --walk-lenght 60 --precompute-rw 4  --epoch 5 --epoch-embedding 1  --epoch-embedding-init 20 --beta .001  --alpha .0005 --gamma .001 --lr  1. --negative-sampling 5  --context-size 5  --cuda --embedding-optimizer exphsgd  --em-iter 5 --id Wikipedia-5D-3  --size 5  --seed 248 --batch-size 512

python3.7 launcher_tools/evaluation_classifier_poincare.py --file /local/gerald/POINCARE-EM/Wikipedia-5D-3

CUDA_VISIBLE_DEVICES=2 python3.7 launcher_tools/experiment_poincare_alternate.py --loss-aggregation sum --distance-coef 1. --dataset wikipedia --n-gaussian 40 --walk-lenght 60 --precompute-rw 4  --epoch 5 --epoch-embedding 1  --epoch-embedding-init 20 --beta .001  --alpha .0005 --gamma .001 --lr  1. --negative-sampling 5  --context-size 5  --cuda --embedding-optimizer exphsgd  --em-iter 5 --id Wikipedia-5D-4  --size 5  --seed 249 --batch-size 512

python3.7 launcher_tools/evaluation_classifier_poincare.py --file /local/gerald/POINCARE-EM/Wikipedia-5D-4

CUDA_VISIBLE_DEVICES=2 python3.7 launcher_tools/experiment_poincare_alternate.py --loss-aggregation sum --distance-coef 1. --dataset wikipedia --n-gaussian 40 --walk-lenght 60 --precompute-rw 4  --epoch 5 --epoch-embedding 1  --epoch-embedding-init 20 --beta .001  --alpha .0005 --gamma .001 --lr  1. --negative-sampling 5  --context-size 5  --cuda --embedding-optimizer exphsgd  --em-iter 5 --id Wikipedia-5D-5  --size 5  --seed 250 --batch-size 512

python3.7 launcher_tools/evaluation_classifier_poincare.py --file /local/gerald/POINCARE-EM/Wikipedia-5D-5