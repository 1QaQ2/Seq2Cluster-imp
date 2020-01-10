#!/bin/bash

cd src

datapath="./dataset/"
dataset="pptw"

test=1
mixture=5
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