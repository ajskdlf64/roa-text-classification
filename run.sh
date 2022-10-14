#!bin/bash
python dataprocess.py --seed 1234 --val_ratio 0.1 --test_ratio 0.1 &&
python train.py --seed 1234 --max_epochs 1 --lr 3e-5 --batch_size 16 --backbone distilbert-base-multilingual-cased &&
python eval.py