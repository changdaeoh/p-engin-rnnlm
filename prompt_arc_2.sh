#!/bin/bash

for ptype in 5 4 3 2 1 0
do
    mod=pythia
    for mod_size in '160m' '6.7b' 
    do
        CUDA_VISIBLE_DEVICES='5' python3 main.py \
            --model hf-causal \
            --model_args pretrained=EleutherAI/${mod}-${mod_size},revision=step100000,dtype="float" \
            --tasks arc_easy \
            --prompt_type $ptype \
            > results_pe/${mod}_${mod_size}_arc_easy_${ptype}.txt
    done

done