#!bin/bash
python dataprocess.py &
python train.py & 
python eval.py