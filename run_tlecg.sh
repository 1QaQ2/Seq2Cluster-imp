#!/bin/bash

cd src

datapath="./dataset/"
dataset="tlecg"

test=1
mixture=3
split=2
encoder_dim=15
gpu=1

echo "************* Seq2Cluster on $dataset ************"
if [ $test = 0 ]; then
        python -u main.py \
                --datapath $datapath \
                --dataset $dataset \
                --nmix $mixture \
                --split $split \
                --endim $encoder_dim
else
        python -u main.py \
                --datapath $datapath \
                --dataset $dataset \
                --nmix $mixture \
                --split $split \
                --endim $encoder_dim \
                --gpu $gpu \
                --test
fi