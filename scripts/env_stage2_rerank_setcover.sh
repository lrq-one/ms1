#!/usr/bin/env bash
source scripts/env_stage1_selector_clean.sh

export LOAD_MODEL_PATH=/path/to/stage1_best.pth

export SELECTOR_LOSS_WEIGHT=0.2

export RERANK_LOSS_WEIGHT=1.0
export RERANK_KL_WEIGHT=0.7
export RERANK_BCE_WEIGHT=0.3
export RERANK_USE_SETCOVER_TEACHER=1
export RERANK_TOPK=256
export RERANK_DETACH_SELECTOR=0

export QUALITY_SETCOVER_TOPK=24
export QUALITY_SETCOVER_LAMBDA_FALSE=0.25
export QUALITY_SETCOVER_LAMBDA_REDUN=0.10

export MODEL_SELECT_METRIC=official_cos
export EPOCHS=4
export LR=8e-5
