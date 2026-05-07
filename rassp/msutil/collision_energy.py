# rassp/msutil/collision_energy.py
import re
import numpy as np

CHARGE_FACTOR = {1: 1.0, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75}

NCE_INSTRUMENTS = {
    "orbitrap",
    "lc-esi-qft",
    "lc-apci-itft",
    "linear ion trap",
    "lc-esi-itft",
    "itft",
}

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def nce_to_ev(nce, precursor_mz, charge=1):
    if precursor_mz is None or not np.isfinite(float(precursor_mz)) or float(precursor_mz) <= 0:
        return None
    cf = CHARGE_FACTOR.get(int(charge), 1.0)
    return float(nce) * float(precursor_mz) / 500.0 * float(cf)

def _norm_inst(instrument):
    return str(instrument or "").strip().lower()

def parse_collision_energy_to_ev(raw_ce, precursor_mz=None, instrument=None, charge=1, numeric_as_nce_for_orbitrap=True):
    """
    Return:
      ce_ev, ce_raw, ce_type, ce_parse_ok

    ce_type:
      eV / keV / V / NCE / percent_NCE / numeric_NCE / numeric_eV / unknown
    """
    if raw_ce is None:
        return None, None, "missing", 0

    raw = str(raw_ce).strip()
    if not raw:
        return None, raw, "missing", 0

    inst = _norm_inst(instrument)
    is_nce_inst = any(x in inst for x in NCE_INSTRUMENTS)

    nums = _FLOAT_RE.findall(raw)
    if not nums:
        return None, raw, "unknown", 0

    val = float(nums[0])
    low = raw.lower()

    if "kev" in low:
        return val * 1000.0, raw, "keV", 1

    # 注意：必须先判断 keV，再判断 eV
    if "ev" in low:
        return val, raw, "eV", 1

    if "(nce)" in low or "nce" in low or "hcd" in low or "%" in low or "nominal" in low:
        ev = nce_to_ev(val, precursor_mz, charge=charge)
        if ev is None:
            return None, raw, "NCE_unresolved", 0
        return ev, raw, "NCE", 1

    # 纯数字：Orbitrap/HCD 默认按 NCE 处理；其他按 eV 处理
    if numeric_as_nce_for_orbitrap and is_nce_inst:
        ev = nce_to_ev(val, precursor_mz, charge=charge)
        if ev is None:
            return None, raw, "numeric_NCE_unresolved", 0
        return ev, raw, "numeric_NCE", 1

    return val, raw, "numeric_eV", 1