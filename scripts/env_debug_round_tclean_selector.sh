#!/usr/bin/env bash
source scripts/clean_lrq_env.sh

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

# ===== 当前合格的小样本 cache =====
export FEAT_CACHE_DIR_TRAIN=/home/lwh/projects/lrq2/ms1/data/nist_20/cache_debug_train_hcd_mh_ce_big100k_round_tclean_bal4k_v4fast
export FEAT_CACHE_DIR_VAL=/home/lwh/projects/lrq2/ms1/data/nist_20/cache_debug_val_hcd_mh_ce_big100k_round_tclean_bal4k_v4fast

# ===== 必须和 cache 生成保持一致 =====
export FORMULA_ATOMICNOS=1,6,7,8,9,14,15,16,17,34,35,53
export MAX_HEAVY_ATOMS=60

export OFFICIAL_BIN_WIDTH=0.01
export OFFICIAL_MAX_MZ=1500.0
export OFFICIAL_BIN_MODE=round
export OFFICIAL_EXCLUDE_PRECURSOR=1
export OFFICIAL_EXCLUDE_PRECURSOR_TOL_DA=0.01
export OFFICIAL_EXCLUDE_PRECURSOR_ISOTOPE_N=2
export OFFICIAL_SPECTRAL_LOSS_MODE=cos

# ===== 条件信息 =====
export NIST20_USE_NCE_AS_MODEL_CE=1
export NIST20_REJECT_INCHIKEY_MISMATCH=1
export DEFAULT_ADDUCT='[M+H]+'
export DEFAULT_INSTRUMENT='Orbitrap'
export DEFAULT_MS_LEVEL=2

# ===== fragment local aux：只当特征，不当 hard mask =====
export USE_FRAGMENT_LOCAL_AUX=1
export GATE_FRAGMENT_LOCAL_AUX=1
export FRAGMENT_LOCAL_AUX_SCALE=1.0
export FRAG_AUX_MAX_DEPTH=2

# ===== 防 target leakage =====
export MODEL_ALLOW_TARGET_ALIGNMENT_FEAT=0
export STRICT_MODEL_KWARG_WHITELIST=1
export DEBUG_TARGET_LEAKAGE_CHECK=0

# ===== selector 必须在完整 4096 上选，不能只用 active mask =====
export USE_FN_FORMULA_LOGITS_AS_SELECTOR=0
export MODEL_TOPK_USE_ACTIVE_MASK=0

# ===== selector / teacher 配置 =====
export SELECTOR_TOPK=256
export MODEL_TOPK_EVAL=256
export TEACHER_TOPK_TRAIN=256
export TEACHER_TOPK_EVAL=512
export TARGET_SUPPORT_TOPK=256

export QUALITY_USE_SETCOVER_TEACHER=1
export QUALITY_SETCOVER_POOL_TOPK=1024
export QUALITY_SETCOVER_TOPK=24
export QUALITY_SETCOVER_LAMBDA_FALSE=0.25
export QUALITY_SETCOVER_LAMBDA_REDUN=0.10
export QUALITY_SETCOVER_HYBRID=1
export QUALITY_SETCOVER_HYBRID_Q6_WEIGHT=0.8
export QUALITY_TARGET_GAMMA=2.0

export SELECTOR_LOSS_WEIGHT=1.0
export SELECTOR_BCE_WEIGHT=0.30
export SELECTOR_KL_WEIGHT=0.70
export SELECTOR_POS_WEIGHT=3.0
export SELECTOR_BALANCED_BCE=1
export SELECTOR_BALANCED_POS_PART=0.7

# ===== 第一阶段先只训 selector，不训谱图头 =====
export RERANK_LOSS_WEIGHT=0.0
export MAIN_CANDIDATE_KL_WEIGHT=0.0
export OFFICIAL_SPECTRAL_LOSS_WEIGHT=0.0
export OFFICIAL_SPECTRAL_KL_WEIGHT=0.0
export FALSE_SUPPORT_LOSS_WEIGHT=0.0
export PEAK_AUX_LOSS_WEIGHT=0.0
export OOS_LOSS_WEIGHT=0.0
export PRECURSOR_LOSS_WEIGHT=0.0
export COARSE_SPECTRAL_AUX_WEIGHT=0.0

# ===== 渲染先关掉，避免还没确认 selector 就引入谱图误差 =====
export FORMULA_RENDER_TOPK=0
export FORMULA_RENDER_TOPK_TRAIN=0
export FORMULA_SCORE_TEMPERATURE=0.70
export FORMULA_SCORE_SCALE=1.0
export FORMULA_ENTROPY_LOSS_WEIGHT=0.0

# ===== 小样本调试参数 =====
export BATCH_SIZE=4
export EPOCHS=2
export LR=1e-4
export NUM_WORKERS=0

# 先短跑，确认没问题后再放大
export MAX_TRAIN_STEPS=100
export MAX_VAL_STEPS=50

export MODEL_SELECT_METRIC=selector_precision_at_256