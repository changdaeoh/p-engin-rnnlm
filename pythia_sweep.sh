#!/bin/bash

mod=pythia
for mod_size in '160m' '410m' '1.4b' '2.8b' '6.7b' '12b'
do

CUDA_VISIBLE_DEVICES='6,7' python3 main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/${mod}-${mod_size},revision=step100000,dtype="float" \
    --tasks winogrande,piqa,arc_challenge,arc_easy,lambada_openai,sciq \
    > results/${mod}_${mod_size}.txt

done