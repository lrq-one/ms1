import torch


def neg_mask_fill_value(x):
    if torch.is_tensor(x) and torch.is_floating_point(x):
        if x.dtype in (torch.float16, torch.bfloat16):
            return -1e4
        try:
            return float(torch.finfo(x.dtype).min)
        except Exception:
            return -1e9
    return -1e9


TARGET_LEAKAGE_KEYS = {
    "spect",
    "spect_raw",

    "true_official_idx",
    "true_official_intensity",
    "true_top20_official_idx",
    "true_top20_official_intensity",
    "true_all_official_idx",
    "true_all_official_intensity",

    "true_precursor_bin",
    "true_precursor_intensity",
    "true_precursor_prob_in_all",
    "true_precursor_present",

    "fragment_node_label",
    "fragment_node_label_top20",
    "fragment_node_true_intensity",
    "fragment_node_true_intensity_share",

    "teacher_formula_probs",
}
