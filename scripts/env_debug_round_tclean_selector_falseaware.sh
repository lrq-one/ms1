#!/usr/bin/env bash
source scripts/env_debug_round_tclean_selector.sh

export DEBUG_TARGET_LEAKAGE_CHECK=0
export TRAIN_SELECTOR_ONLY_STAGE=1

# selector 最终先用 64，不再用 256
export SELECTOR_TOPK=64
export MODEL_TOPK_EVAL=64
export TARGET_SUPPORT_TOPK=64
export TEACHER_TOPK_TRAIN=128
export TEACHER_TOPK_EVAL=128

# frag aux 降权，避免捷径
export FRAGMENT_LOCAL_AUX_SCALE=0.2

# selector 仍然主导
export SELECTOR_LOSS_WEIGHT=1.0
export SELECTOR_BCE_WEIGHT=0.4
export SELECTOR_KL_WEIGHT=0.6
export SELECTOR_POS_WEIGHT=3.0
export SELECTOR_BALANCED_BCE=1

# 关键：给 selected set 的 false support 加约束
export FALSE_SUPPORT_LOSS_WEIGHT=0.05

# 不开完整谱图头，避免混乱
export OFFICIAL_SPECTRAL_LOSS_WEIGHT=0.0
export OFFICIAL_SPECTRAL_KL_WEIGHT=0.0
export MAIN_CANDIDATE_KL_WEIGHT=0.0
export RERANK_LOSS_WEIGHT=0.0
export PEAK_AUX_LOSS_WEIGHT=0.0
export OOS_LOSS_WEIGHT=0.0
export PRECURSOR_LOSS_WEIGHT=0.0
export COARSE_SPECTRAL_AUX_WEIGHT=0.0

# 让 false-support loss 可以基于 selected candidates 渲染
export FORMULA_RENDER_TOPK=64
export FORMULA_RENDER_TOPK_TRAIN=64

export EVAL_TOPK_LIST=32,64,128,256

export BATCH_SIZE=4
export NUM_WORKERS=0
export EPOCHS=3
export MAX_TRAIN_STEPS=800
export MAX_VAL_STEPS=100

export MODEL_SELECT_METRIC=official_cos