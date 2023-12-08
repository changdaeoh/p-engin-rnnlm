#!/bin/bash

mod=rwkv-4
for mod_size in '169m' '430m' '1b5' '3b' '7b' '14b'
do

CUDA_VISIBLE_DEVICES='2,3' python3 main.py \
    --model hf-causal \
    --model_args pretrained=RWKV/${mod}-${mod_size}-pile \
    --tasks winogrande,piqa,arc_challenge,arc_easy,lambada_openai,lambada_openai_cloze,lambada_standard,lambada_standard_cloze,sciq \
    > results/${mod}_${mod_size}.txt

done