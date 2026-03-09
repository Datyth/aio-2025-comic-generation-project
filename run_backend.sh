#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=2

PYTHONPATH=./backend uvicorn backend.main:app --reload --port 8000