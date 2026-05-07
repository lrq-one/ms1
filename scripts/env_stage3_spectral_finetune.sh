#!/usr/bin/env bash
source scripts/env_stage2_rerank_setcover.sh

export LOAD_MODEL_PATH=/path/to/stage2_best.pth

export FORMULA_RENDER_TOPK=64
export FORMULA_RENDER_TOPK_TRAIN=1

export SELECTOR_LOSS_WEIGHT=0.1
export RERANK_LOSS_WEIGHT=0.5

export MAIN_CANDIDATE_KL_WEIGHT=0.02
export OFFICIAL_SPECTRAL_LOSS_WEIGHT=0.003
export FALSE_SUPPORT_LOSS_WEIGHT=0.001

export LR=3e-5
export EPOCHS=2
export MODEL_SELECT_METRIC=official_cos
