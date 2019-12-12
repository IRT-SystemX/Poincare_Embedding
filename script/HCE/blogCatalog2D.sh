CUDA_VISIBLE_DEVICES=0 python3.7 launcher_tools/experiment_poincare_alternate_moving_context_size.py --loss-aggregation sum --distance-coef 1.0  --dataset blogCatalog  --n-gaussian 5 --walk-lenght 80 --precompute-rw 10  --epoch 5 --epoch-embedding 2  --epoch-embedding-init 20 --beta 0.1  --alpha 1 --gamma 1. --lr  .1 --negative-sampling 5  --context-size 10  --cuda --embedding-optimizer exphsgd  --em-iter 5 --id BLOG-TEST1 --size 5  --seed 300 --batch-size 128


python3.7 launcher_tools/evaluation_unsupervised_poincare.py --file /local/gerald/POINCARE-EM/BLOG-TEST1
python3.7 launcher_tools/evaluation_classifier_poincare.py --init --file /local/gerald/POINCARE-EM/BLOG-TEST1
python3.7 launcher_tools/evaluation_supervised_poincare.py --file /local/gerald/POINCARE-EM/BLOG-TEST1

# CUDA_VISIBLE_DEVICES=0 python3.7 launcher_tools/experiment_poincare_alternate_moving_context_size.py --loss-aggregation sum --distance-coef 1.0 --dataset blogCatalog  --n-gaussian 39 --walk-lenght 60 --precompute-rw 10  --epoch 5 --epoch-embedding 1  --epoch-embedding-init 1 --beta 1.  --alpha 1. --gamma 1. --lr  .01 --negative-sampling 5  --context-size 10  --cuda --embedding-optimizer exphsgd  --em-iter 5 --id 12102019blogCatalog-2D-2  --size 2 --seed 247 --batch-size 64

# python3.7 launcher_tools/evaluation_classifier_poincare.py --init --file /local/gerald/POINCARE-EM/12102019blogCatalog-2D-2
# python3.7 launcher_tools/evaluation_supervised_poincare.py --file /local/gerald/POINCARE-EM/12102019blogCatalog-2D-2

# CUDA_VISIBLE_DEVICES=0 python3.7 launcher_tools/experiment_poincare_alternate_moving_context_size.py --loss-aggregation sum --distance-coef 1.0 --dataset blogCatalog  --n-gaussian 39 --walk-lenght 60 --precompute-rw 10  --epoch 5 --epoch-embedding 1  --epoch-embedding-init 1 --beta 1.  --alpha 1. --gamma 1. --lr  .01 --negative-sampling 5  --context-size 10  --cuda --embedding-optimizer exphsgd  --em-iter 5 --id 12102019blogCatalog-2D-3  --size 2 --seed 248 --batch-size 64

# python3.7 launcher_tools/evaluation_classifier_poincare.py --init --file /local/gerald/POINCARE-EM/12102019blogCatalog-2D-3
# python3.7 launcher_tools/evaluation_supervised_poincare.py --file /local/gerald/POINCARE-EM/12102019blogCatalog-2D-3

# CUDA_VISIBLE_DEVICES=0 python3.7 launcher_tools/experiment_poincare_alternate_moving_context_size.py --loss-aggregation sum --distance-coef 1.0 --dataset blogCatalog  --n-gaussian 39 --walk-lenght 60 --precompute-rw 10  --epoch 5 --epoch-embedding 1  --epoch-embedding-init 1 --beta 1.  --alpha 1. --gamma 1. --lr  .01 --negative-sampling 5  --context-size 10  --cuda --embedding-optimizer exphsgd  --em-iter 5 --id 12102019blogCatalog-2D-4  --size 2 --seed 249 --batch-size 64

# python3.7 launcher_tools/evaluation_classifier_poincare.py --init --file /local/gerald/POINCARE-EM/12102019blogCatalog-2D-4
# python3.7 launcher_tools/evaluation_supervised_poincare.py --file /local/gerald/POINCARE-EM/12102019blogCatalog-2D-4

# CUDA_VISIBLE_DEVICES=0 python3.7 launcher_tools/experiment_poincare_alternate_moving_context_size.py --loss-aggregation sum --distance-coef 1.0 --dataset blogCatalog  --n-gaussian 39 --walk-lenght 60 --precompute-rw 10  --epoch 5 --epoch-embedding 1  --epoch-embedding-init 1 --beta 1.  --alpha 1. --gamma 1. --lr  .01 --negative-sampling 5  --context-size 10  --cuda --embedding-optimizer exphsgd  --em-iter 5 --id 12102019blogCatalog-2D-5  --size 2 --seed 250 --batch-size 64

# python3.7 launcher_tools/evaluation_classifier_poincare.py --init --file /local/gerald/POINCARE-EM/12102019blogCatalog-2D-5
# python3.7 launcher_tools/evaluation_supervised_poincare.py --file /local/gerald/POINCARE-EM/12102019blogCatalog-2D-5