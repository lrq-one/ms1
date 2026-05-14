import os
from dataclasses import dataclass


def env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def env_bool(name, default=False):
    raw = os.environ.get(name, "1" if default else "0")
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class RunConfig:
    epochs: int
    max_train_steps: int
    max_val_steps: int
    batch_size: int
    model_select_metric: str


@dataclass
class SelectorConfig:
    selector_topk: int
    model_topk_eval: int
    target_support_topk: int
    use_coverage_aware_topk: bool
    coverage_dup_penalty: float
    coverage_novelty_bonus: float
    use_group_unique_model: bool
    use_group_unique_teacher: bool
    use_group_unique_prune: bool


@dataclass
class LossConfig:
    selector_weight: float
    selector_bce_weight: float
    selector_kl_weight: float
    selector_pos_weight: float
    selector_utility_weight: float
    selector_pairwise_weight: float
    selector_recall_bce_weight: float
    selector_false_lambda: float
    false_support_weight: float
    soft_false_support_weight: float
    rerank_weight: float
    official_spectral_weight: float
    peak_weight: float
    oos_weight: float
    precursor_weight: float


def get_run_config():
    return RunConfig(
        epochs=env_int("EPOCHS", 5),
        max_train_steps=env_int("MAX_TRAIN_STEPS", 0),
        max_val_steps=env_int("MAX_VAL_STEPS", 0),
        batch_size=env_int("BATCH_SIZE", 4),
        model_select_metric=os.environ.get("MODEL_SELECT_METRIC", "model_topk_oracle_cos_64"),
    )


def get_selector_config():
    return SelectorConfig(
        selector_topk=env_int("SELECTOR_TOPK", 64),
        model_topk_eval=env_int("MODEL_TOPK_EVAL", 64),
        target_support_topk=env_int("TARGET_SUPPORT_TOPK", 64),
        use_coverage_aware_topk=env_bool("USE_COVERAGE_AWARE_TOPK", False),
        coverage_dup_penalty=env_float("COVERAGE_TOPK_DUP_PENALTY", 0.35),
        coverage_novelty_bonus=env_float("COVERAGE_TOPK_NOVELTY_BONUS", 0.10),
        use_group_unique_model=env_bool("USE_GROUP_UNIQUE_MODEL", False),
        use_group_unique_teacher=env_bool("USE_GROUP_UNIQUE_TEACHER", False),
        use_group_unique_prune=env_bool("USE_GROUP_UNIQUE_PRUNE", False),
    )


def get_loss_config():
    return LossConfig(
        selector_weight=env_float("SELECTOR_LOSS_WEIGHT", 1.0),
        selector_bce_weight=env_float("SELECTOR_BCE_WEIGHT", 0.15),
        selector_kl_weight=env_float("SELECTOR_KL_WEIGHT", 0.15),
        selector_pos_weight=env_float("SELECTOR_POS_WEIGHT", 5.0),
        selector_utility_weight=env_float("SELECTOR_UTILITY_LOSS_WEIGHT", 0.50),
        selector_pairwise_weight=env_float("SELECTOR_PAIRWISE_WEIGHT", 0.20),
        selector_recall_bce_weight=env_float("SELECTOR_RECALL_BCE_WEIGHT", 0.20),
        selector_false_lambda=env_float("SELECTOR_UTILITY_FALSE_LAMBDA", 0.60),
        false_support_weight=env_float("FALSE_SUPPORT_LOSS_WEIGHT", 0.20),
        soft_false_support_weight=env_float(
            "SOFT_FALSE_SUPPORT_WEIGHT",
            env_float("SOFT_FALSE_SUPPORT_LOSS_WEIGHT", 0.0),
        ),
        rerank_weight=env_float("RERANK_LOSS_WEIGHT", 0.0),
        official_spectral_weight=env_float("OFFICIAL_SPECTRAL_LOSS_WEIGHT", 0.0),
        peak_weight=env_float("PEAK_LOSS_WEIGHT", 0.0),
        oos_weight=env_float("OOS_LOSS_WEIGHT", 0.0),
        precursor_weight=env_float("PRECURSOR_LOSS_WEIGHT", 0.0),
    )
