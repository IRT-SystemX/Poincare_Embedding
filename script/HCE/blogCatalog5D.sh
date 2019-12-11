# CUDA_VISIBLE_DEVICES=3 python3.7 launcher_tools/experiment_poincare_alternate.py --loss-aggregation sum --distance-coef 0.7 --dataset blogCatalog  --n-gaussian 39 --walk-lenght 60 --precompute-rw 8  --epoch 5 --epoch-embedding 1  --epoch-embedding-init 10 --beta 1.  --alpha .5 --gamma 10. --lr  .002 --negative-sampling 5  --context-size 5  --cuda --embedding-optimizer exphsgd  --em-iter 5 --id 12092019blogCatalog-5D-1  --size 5  --seed 246 --batch-size 256

# python3.7 launcher_tools/evaluation_unsupervised_poincare.py --file /local/gerald/POINCARE-EM/12092019blogCatalog-5D-1
# python3.7 launcher_tools/evaluation_classifier_poincare.py --init --file /local/gerald/POINCARE-EM/12092019blogCatalog-5D-1
python3.7 launcher_tools/evaluation_supervised_poincare.py --file /local/gerald/POINCARE-EM/12092019blogCatalog-5D-1 --init

# # CUDA_VISIBLE_DEVICES=3 python3.7 launcher_tools/experiment_poincare_alternate.py --loss-aggregation sum --distance-coef 0.7 --dataset blogCatalog  --n-gaussian 39 --walk-lenght 60 --precompute-rw 4  --epoch 5 --epoch-embedding 1  --epoch-embedding-init 10 --beta 1.  --alpha .5 --gamma 10. --lr  .002 --negative-sampling 5  --context-size 5  --cuda --embedding-optimizer exphsgd  --em-iter 5 --id 12092019blogCatalog-5D-2  --size 5  --seed 247 --batch-size 256

# python3.7 launcher_tools/evaluation_classifier_poincare.py --init --file /local/gerald/POINCARE-EM/12092019blogCatalog-5D-2
# # python3.7 launcher_tools/evaluation_supervised_poincare.py --file /local/gerald/POINCARE-EM/12092019blogCatalog-5D-2

# # CUDA_VISIBLE_DEVICES=3 python3.7 launcher_tools/experiment_poincare_alternate.py --loss-aggregation sum --distance-coef 0.7 --dataset blogCatalog  --n-gaussian 39 --walk-lenght 60 --precompute-rw 4  --epoch 5 --epoch-embedding 1  --epoch-embedding-init 10 --beta 1.  --alpha .5 --gamma 10. --lr  .002 --negative-sampling 5  --context-size 5  --cuda --embedding-optimizer exphsgd  --em-iter 5 --id 12092019blogCatalog-5D-3  --size 5  --seed 248 --batch-size 256

# python3.7 launcher_tools/evaluation_classifier_poincare.py --init --file /local/gerald/POINCARE-EM/12092019blogCatalog-5D-3
# # python3.7 launcher_tools/evaluation_supervised_poincare.py --file /local/gerald/POINCARE-EM/12092019blogCatalog-5D-3

# # CUDA_VISIBLE_DEVICES=3 python3.7 launcher_tools/experiment_poincare_alternate.py --loss-aggregation sum --distance-coef 0.7 --dataset blogCatalog  --n-gaussian 39 --walk-lenght 60 --precompute-rw 4  --epoch 5 --epoch-embedding 1  --epoch-embedding-init 10 --beta 1.  --alpha .5 --gamma 10. --lr  .002 --negative-sampling 5  --context-size 5  --cuda --embedding-optimizer exphsgd  --em-iter 5 --id 12092019blogCatalog-5D-4  --size 5  --seed 249 --batch-size 256

# python3.7 launcher_tools/evaluation_classifier_poincare.py --init --file /local/gerald/POINCARE-EM/12092019blogCatalog-5D-4
# # python3.7 launcher_tools/evaluation_supervised_poincare.py --file /local/gerald/POINCARE-EM/12092019blogCatalog-5D-4

# # CUDA_VISIBLE_DEVICES=3 python3.7 launcher_tools/experiment_poincare_alternate.py --loss-aggregation sum --distance-coef 0.7 --dataset blogCatalog  --n-gaussian 39 --walk-lenght 60 --precompute-rw 4  --epoch 5 --epoch-embedding 1  --epoch-embedding-init 10 --beta 1.  --alpha .5 --gamma 10. --lr  .002 --negative-sampling 5  --context-size 5  --cuda --embedding-optimizer exphsgd  --em-iter 5 --id 12092019blogCatalog-5D-5  --size 5  --seed 250 --batch-size 256

# python3.7 launcher_tools/evaluation_classifier_poincare.py --init --file /local/gerald/POINCARE-EM/12092019blogCatalog-5D-5
# # python3.7 launcher_tools/evaluation_supervised_poincare.py --file /local/gerald/POINCARE-EM/12092019blogCatalog-5D-5