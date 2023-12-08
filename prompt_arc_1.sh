#!/bin/bash

for ptype in 0 1 2 3 4 5
do
    mod=rwkv-4
    for mod_size in '169m' '7b' 
    do
        CUDA_VISIBLE_DEVICES='4' python3 main.py \
            --model hf-causal \
            --model_args pretrained=RWKV/${mod}-${mod_size}-pile \
            --tasks arc_easy \
            --prompt_type $ptype \
            > results_pe/${mod}_${mod_size}_arc_easy_${ptype}.txt
    done

done