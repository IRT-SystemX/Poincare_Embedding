CUDA_VISIBLE_DEVICES=2 python3.7 launcher_tools/experiment_poincare_alternate.py --loss-aggregation sum --distance-coef 1.0  --dataset dblp  --n-gaussian 5 --walk-lenght 60 --precompute-rw 8  --epoch 5 --epoch-embedding 1  --epoch-embedding-init 10 --beta 1.  --alpha .5 --gamma 10. --lr  1e-4 --negative-sampling 5  --context-size 5  --cuda --embedding-optimizer exphsgd  --em-iter 5 --id 11122019DBLP-2D-1  --size 2  --seed 246 --batch-size 8192

python3.7 launcher_tools/evaluation_classifier_poincare.py --file /local/gerald/POINCARE-EM/11122019DBLP-2D-1
python3.7 launcher_tools/evaluation_supervised_poincare.py --file /local/gerald/POINCARE-EM/11122019DBLP-2D-1

CUDA_VISIBLE_DEVICES=2 python3.7 launcher_tools/experiment_poincare_alternate.py --loss-aggregation sum --distance-coef 1.0  --dataset dblp  --n-gaussian 5 --walk-lenght 60 --precompute-rw 8  --epoch 5 --epoch-embedding 1  --epoch-embedding-init 10 --beta 1.  --alpha .5 --gamma 10. --lr  1e-4 --negative-sampling 5  --context-size 5  --cuda --embedding-optimizer exphsgd  --em-iter 5 --id 11122019DBLP-2D-2  --size 2  --seed 247 --batch-size 8192

python3.7 launcher_tools/evaluation_classifier_poincare.py --file /local/gerald/POINCARE-EM/11122019DBLP-2D-2
python3.7 launcher_tools/evaluation_supervised_poincare.py --file /local/gerald/POINCARE-EM/11122019DBLP-2D-2

CUDA_VISIBLE_DEVICES=2 python3.7 launcher_tools/experiment_poincare_alternate.py --loss-aggregation sum --distance-coef 1.0  --dataset dblp  --n-gaussian 5 --walk-lenght 60 --precompute-rw 8  --epoch 5 --epoch-embedding 1  --epoch-embedding-init 10 --beta 1.  --alpha .5 --gamma 10. --lr  1e-4 --negative-sampling 5  --context-size 5  --cuda --embedding-optimizer exphsgd  --em-iter 5 --id 11122019DBLP-2D-3  --size 2  --seed 248 --batch-size 8192

python3.7 launcher_tools/evaluation_classifier_poincare.py --file /local/gerald/POINCARE-EM/11122019DBLP-2D-3
python3.7 launcher_tools/evaluation_supervised_poincare.py --file /local/gerald/POINCARE-EM/11122019DBLP-2D-3

CUDA_VISIBLE_DEVICES=2 python3.7 launcher_tools/experiment_poincare_alternate.py --loss-aggregation sum --distance-coef 1.0  --dataset dblp  --n-gaussian 5 --walk-lenght 60 --precompute-rw 8  --epoch 5 --epoch-embedding 1  --epoch-embedding-init 10 --beta 1.  --alpha .5 --gamma 10. --lr  1e-4 --negative-sampling 5  --context-size 5  --cuda --embedding-optimizer exphsgd  --em-iter 5 --id 11122019DBLP-2D-4  --size 2  --seed 249 --batch-size 8192

python3.7 launcher_tools/evaluation_classifier_poincare.py --file /local/gerald/POINCARE-EM/11122019DBLP-2D-4
python3.7 launcher_tools/evaluation_supervised_poincare.py --file /local/gerald/POINCARE-EM/11122019DBLP-2D-4

CUDA_VISIBLE_DEVICES=2 python3.7 launcher_tools/experiment_poincare_alternate.py --loss-aggregation sum --distance-coef 1.0  --dataset dblp  --n-gaussian 5 --walk-lenght 60 --precompute-rw 8  --epoch 5 --epoch-embedding 1  --epoch-embedding-init 10 --beta 1.  --alpha .5 --gamma 10. --lr  1e-4 --negative-sampling 5  --context-size 5  --cuda --embedding-optimizer exphsgd  --em-iter 5 --id 11122019DBLP-2D-5  --size 2  --seed 250 --batch-size 8192

python3.7 launcher_tools/evaluation_classifier_poincare.py --file /local/gerald/POINCARE-EM/11122019DBLP-2D-5
python3.7 launcher_tools/evaluation_supervised_poincare.py --file /local/gerald/POINCARE-EM/11122019DBLP-2D-5