#!/bin/bash

mod=bloom
for mod_size in '1b7'
do

CUDA_VISIBLE_DEVICES='4,5' python3 main.py > results/${mod}_${mod_size}.txt \
    --model hf-causal \
    --model_args pretrained=bigscience/${mod}-${mod_size} \
    --tasks winogrande,piqa,arc_challenge,arc_easy,lambada_openai,sciq

done

mod=bloomz
for mod_size in '3b'
do

CUDA_VISIBLE_DEVICES='4,5' python3 main.py > results/${mod}_${mod_size}.txt \
    --model hf-causal \
    --model_args pretrained=bigscience/${mod}-${mod_size} \
    --tasks winogrande,piqa,arc_challenge,arc_easy,lambada_openai,sciq

done