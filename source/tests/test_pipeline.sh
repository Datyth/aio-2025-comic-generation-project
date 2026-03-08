#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=2

python ./source/tests/test_pipeline.py