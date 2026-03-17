#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=2

cd "$(dirname "$0")" || exit
PYTHONPATH=./backend uvicorn backend.main:app --host 0.0.0.0 --reload --port 8000