#!/bin/bash

for ptype in 0 1 2 3 4 5
do
    mod=pythia
    for mod_size in '160m' '6.7b' 
    do
        CUDA_VISIBLE_DEVICES='3' python3 main.py \
            --model hf-causal \
            --model_args pretrained=EleutherAI/${mod}-${mod_size},revision=step100000,dtype="float" \
            --tasks sciq \
            --prompt_type $ptype \
            > results_pe/${mod}_${mod_size}_sciq_${ptype}.txt
    done

done