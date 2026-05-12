# # # # import os, glob

# # # # MOL_PATH = "/home/lwh/projects/lrq2/ms1/data/nist_20/hr_msms_nist.MOL"
# # # # MSP_PATH = "/home/lwh/projects/lrq2/ms1/data/nist_20/hr_msms_nist.MSP"

# # # # print("MOL exists:", os.path.exists(MOL_PATH))
# # # # print("MOL is dir:", os.path.isdir(MOL_PATH))
# # # # print("MOL is file:", os.path.isfile(MOL_PATH))
# # # # print("MSP exists:", os.path.exists(MSP_PATH))
# # # # print("MSP is file:", os.path.isfile(MSP_PATH))

# # # # if os.path.isdir(MOL_PATH):
# # # #     files = sorted(glob.glob(os.path.join(MOL_PATH, "*.MOL")))
# # # #     print("MOL file count:", len(files))
# # # #     print("first 10 MOL files:", [os.path.basename(x) for x in files[:10]])
# # # # else:
# # # #     print("MOL size:", os.path.getsize(MOL_PATH) if os.path.exists(MOL_PATH) else None)


# # # from collections import Counter
# # # import re, os

# # # MSP_PATH = "/home/lwh/projects/lrq2/ms1/data/nist_20/hr_msms_nist.MSP"

# # # field_counter = Counter()
# # # name_n = 0
# # # peak_block_n = 0
# # # id_n = 0
# # # first_blocks = []

# # # cur = []
# # # with open(MSP_PATH, "r", encoding="utf-8", errors="replace") as f:
# # #     for line in f:
# # #         line = line.rstrip("\n\r")
# # #         if line.startswith("Name:"):
# # #             if cur and len(first_blocks) < 3:
# # #                 first_blocks.append(cur)
# # #             cur = [line]
# # #             name_n += 1
# # #             continue

# # #         if cur is not None and len(first_blocks) < 3:
# # #             cur.append(line)

# # #         if ":" in line:
# # #             k = line.split(":", 1)[0].strip()
# # #             field_counter[k] += 1
# # #             if k.lower() == "id":
# # #                 id_n += 1
# # #             if k.lower() == "num peaks":
# # #                 peak_block_n += 1

# # # if cur and len(first_blocks) < 3:
# # #     first_blocks.append(cur)

# # # print("Name blocks:", name_n)
# # # print("ID fields:", id_n)
# # # print("Num peaks fields:", peak_block_n)
# # # print("Top fields:")
# # # for k, v in field_counter.most_common(30):
# # #     print(k, v)

# # # print("\n==== first 3 blocks preview ====")
# # # for i, b in enumerate(first_blocks):
# # #     print("\n--- BLOCK", i, "---")
# # #     print("\n".join(b[:40]))


# # import os, sys
# # sys.path.insert(0, "/home/lwh/projects/lrq2/ms1")  # 改成你的仓库路径

# # from rassp.data.nist20_reader import load_nist20_records
# # from collections import Counter
# # import numpy as np

# # MOL_DIR = "/home/lwh/projects/lrq2/ms1/data/nist_20/hr_msms_nist.MOL"
# # MSP_PATH = "/home/lwh/projects/lrq2/ms1/data/nist_20/hr_msms_nist.MSP"

# # records = load_nist20_records(
# #     mol_dir=MOL_DIR,
# #     msp_path=MSP_PATH,
# #     split=None,
# #     seed=42,
# #     split_ratios="0.8,0.1,0.1",
# #     limit=5000,
# #     require_ce=0,
# #     allowed_adducts=None,
# #     allowed_instruments=None,
# #     allowed_fragmentation_methods=None,
# #     max_precursor_mz=0.0,
# #     progress_every=1000,
# # )

# # print("loaded records:", len(records))

# # def count(key):
# #     return Counter(str(r.get(key, "<missing>")) for r in records)

# # print("adduct:", count("adduct").most_common(20))
# # print("instrument_type/platform:", count("instrument_type").most_common(20))
# # print("fragmentation_method:", count("fragmentation_method").most_common(20))
# # print("collision_energy_type:", count("collision_energy_type").most_common(20))
# # print("collision_energy_parse_ok:", count("collision_energy_parse_ok").most_common())
# # print("soft_name_match:", count("soft_name_match").most_common())
# # print("soft_inchikey_match:", count("soft_inchikey_match").most_common())

# # peak_ns = []
# # mz_maxs = []
# # int_sums = []
# # precursors = []
# # for r in records:
# #     s = np.asarray(r["spect"], dtype=np.float32)
# #     peak_ns.append(s.shape[0])
# #     mz_maxs.append(float(np.max(s[:, 0])) if s.size else np.nan)
# #     int_sums.append(float(np.sum(s[:, 1])) if s.size else np.nan)
# #     if r.get("precursor_mz") is not None:
# #         precursors.append(float(r["precursor_mz"]))

# # for name, arr in [
# #     ("peak_n", peak_ns),
# #     ("mz_max", mz_maxs),
# #     ("int_sum", int_sums),
# #     ("precursor_mz", precursors),
# # ]:
# #     arr = np.asarray(arr, dtype=np.float64)
# #     arr = arr[np.isfinite(arr)]
# #     print(name, {
# #         "n": int(arr.size),
# #         "mean": float(arr.mean()) if arr.size else None,
# #         "p50": float(np.percentile(arr, 50)) if arr.size else None,
# #         "p90": float(np.percentile(arr, 90)) if arr.size else None,
# #         "max": float(arr.max()) if arr.size else None,
# #     })

# # print("\nfirst record keys:", records[0].keys())
# # print("first identifier:", records[0].get("identifier"))
# # print("first smiles:", records[0].get("smiles"))
# # print("first adduct:", records[0].get("adduct"))
# # print("first instrument_type:", records[0].get("instrument_type"))
# # print("first fragmentation_method:", records[0].get("fragmentation_method"))
# # print("first CE:", records[0].get("collision_energy"), records[0].get("collision_energy_raw"), records[0].get("collision_energy_type"))
# # print("first precursor_mz:", records[0].get("precursor_mz"))
# # print("first spect shape:", np.asarray(records[0]["spect"]).shape)
# # print("first 10 peaks:", np.asarray(records[0]["spect"])[:10])





# import sys, numpy as np, pandas as pd
# sys.path.insert(0, "/home/lwh/projects/lrq2/ms1") 

# from rassp.data.nist20_reader import load_nist20_records, records_to_dataframes

# MOL_DIR = "/home/lwh/projects/lrq2/ms1/data/nist_20/hr_msms_nist.MOL"
# MSP_PATH = "/home/lwh/projects/lrq2/ms1/data/nist_20/hr_msms_nist.MSP"

# records = load_nist20_records(
#     MOL_DIR,
#     MSP_PATH,
#     split="train",
#     seed=42,
#     split_ratios="0.8,0.1,0.1",
#     limit=1000,
#     require_ce=0,
# )

# df_rows, tsv_rows = records_to_dataframes(records, split="train")

# print("records:", len(records))
# print("df_rows:", len(df_rows))
# print("tsv_rows:", len(tsv_rows))

# bad = []
# for i, (d, t) in enumerate(zip(df_rows, tsv_rows)):
#     s = np.asarray(d["spect"], dtype=np.float32)
#     mzs = [x for x in str(t["mzs"]).split(",") if x]
#     ints = [x for x in str(t["intensities"]).split(",") if x]
#     if s.shape[0] != len(mzs) or s.shape[0] != len(ints):
#         bad.append((i, "len", s.shape[0], len(mzs), len(ints)))
#         continue
#     if s.shape[0] > 0:
#         if abs(float(s[0,0]) - float(mzs[0])) > 1e-4:
#             bad.append((i, "mz0", s[0,0], mzs[0]))
#         if abs(float(s[0,1]) - float(ints[0])) > 1e-6:
#             bad.append((i, "int0", s[0,1], ints[0]))

# print("bad alignment count:", len(bad))
# print("first bad:", bad[:5])

# print("first df keys:", df_rows[0].keys())
# print("first tsv keys:", tsv_rows[0].keys())
# print("first tsv:", tsv_rows[0])

import os, glob, pickle
import numpy as np

CACHE_DIR = "/home/lwh/projects/lrq2/ms1/data/nist_20/cache_train3000_v3c_balanced_recursive_fragment_node"
paths = sorted(glob.glob(os.path.join(CACHE_DIR, "*.pkl")))[:200]

bad = []
for p in paths:
    obj = pickle.load(open(p, "rb"))
    feat = obj.get("features", {})

    mask = np.asarray(feat.get("formulae_mask", []), dtype=np.float32).reshape(-1)
    off_idx = np.asarray(feat.get("formulae_peaks_official_idx", []), dtype=np.int64)
    off_int = np.asarray(feat.get("formulae_peaks_official_intensity", []), dtype=np.float32)

    if mask.size == 0 or off_idx.ndim != 2 or off_int.ndim != 2:
        bad.append((os.path.basename(p), "shape", mask.shape, off_idx.shape, off_int.shape))
        continue

    n = min(mask.shape[0], off_idx.shape[0], off_int.shape[0])
    mask = mask[:n]
    off_idx = off_idx[:n]
    off_int = off_int[:n]

    invalid = mask <= 0.5
    if invalid.any():
        invalid_idx_bad = np.any(off_idx[invalid] >= 0)
        invalid_int_bad = np.any(off_int[invalid] > 0)
        if invalid_idx_bad or invalid_int_bad:
            bad.append((
                os.path.basename(p),
                "invalid_candidate_has_peak",
                int(invalid.sum()),
                bool(invalid_idx_bad),
                bool(invalid_int_bad),
            ))

print("checked:", len(paths))
print("bad:", len(bad))
print("first bad:", bad[:10])