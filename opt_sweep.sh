#!/bin/bash

mod=opt
for mod_size in '13b' '6.7b' '2.7b' '1.3b' '350m' '125m'
do

CUDA_VISIBLE_DEVICES='0,1' python3 main.py \
    --model hf-causal \
    --model_args pretrained=facebook/${mod}-${mod_size} \
    --tasks winogrande,piqa,arc_challenge,arc_easy,lambada_openai,sciq \
    > results/${mod}_${mod_size}.txt

done