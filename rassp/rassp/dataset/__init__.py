# File overview: This module is part of the MassSpecGym/RASSP codebase.
# Purpose: Dataset loading and sample assembly logic for training and evaluation data sources.

import numpy as np
import pandas as pd

import torch.utils.data
import pickle
from rdkit import Chem
from sqlalchemy import create_engine, MetaData, select, Table
import os

from rassp.featurize import create_mol_featurizer, create_pred_featurizer

# SQLite-backed dataset loader used for legacy database training/evaluation paths.
class DBDataset:
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(
        self,
        db_filename,
        db_id_field,
        db_ids,
        spect_bin_config,
        featurizer_config,
        pred_featurizer_config,
    ):
        self.db_filename = db_filename
        self.db_id_field = db_id_field
        self.db_ids = db_ids

        self.featurizer = create_mol_featurizer(spect_bin_config, featurizer_config)
        self.pred_featurizer = create_pred_featurizer(spect_bin_config, pred_featurizer_config)

    # Create SQLAlchemy engine for on-demand row reads.
    def create_db_engine(self, db_filename):
        assert os.path.exists(db_filename)
        engine = create_engine(f"sqlite+pysqlite:///{db_filename}", future=True)
        return engine
    
    # Function overview: __len__ handles a specific workflow step in this module.
    def __len__(self):
        return len(self.db_ids)
    
    # Fetch one DB row, featurize molecule, and return model-ready dict.
    def __getitem__(self, idx):
        
        engine = self.create_db_engine(self.db_filename)
        metadata_obj = MetaData()

        mol_table = Table("molecules", metadata_obj, autoload_with=engine)

        tgt_id = self.db_ids[idx]

        # load from database
        with engine.connect() as conn:
            table = mol_table
            stmt = select([table.c.bmol, table.c.num_spectrum_peaks, table.c.binary_spectrum]).where(table.c[self.db_id_field] == tgt_id)
            
            db_record = conn.execute(stmt).one()
            
        mol = Chem.Mol(db_record['bmol'])
        num_peaks = db_record['num_spectrum_peaks']
        sparse_spect_shape = (num_peaks, 2)

        spect_sparse = np.frombuffer(db_record['binary_spectrum'],
                                     dtype=np.float32).reshape(sparse_spect_shape)

        features_dict = self.featurizer(mol, spect_sparse)
        preds_dict = self.pred_featurizer(mol, spect_sparse)
        out_dict = {
            **features_dict,
            **preds_dict,
        }
        out_dict['input_idx'] = idx

        return out_dict

# Parquet-backed dataset with optional per-row feature cache and condition imputation.
class ParquetDataset:
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(
        self,
        filename,
        spect_bin_config,
        featurizer_config,
        pred_featurizer_config,
    ):
        def _parse_csv_env(raw):
            if raw is None:
                return []
            out = []
            for token in str(raw).split(','):
                token = token.strip()
                if token:
                    out.append(token)
            return out

        self.filename = filename
        # optional feature cache directory (per-row pickle files)
        # default: <parquet_filename>.cache or override with env FEAT_CACHE_DIR
        cache_dir_candidate = os.environ.get('FEAT_CACHE_DIR', filename + '.cache')
        if os.path.exists(cache_dir_candidate) and os.path.isdir(cache_dir_candidate):
            # Prefer using the cache if it exists to avoid needing a parquet engine
            self.cache_dir = cache_dir_candidate
            # infer number of rows from cache file names like "0.pkl", "1.pkl" ...
            try:
                names = os.listdir(self.cache_dir)
                idxs = []
                for n in names:
                    if n.endswith('.pkl'):
                        try:
                            idxs.append(int(os.path.splitext(n)[0]))
                        except Exception:
                            continue
                if len(idxs) == 0:
                    # empty cache directory — fall back to parquet
                    self.cache_dir = None
                    self.df = pd.read_parquet(filename)
                else:
                    max_idx = max(idxs)
                    # create a lightweight placeholder dataframe compatible with existing code
                    nrows = max_idx + 1
                    # before trusting the cache, ensure cached entries include required feature keys
                    try:
                        sample_idx = min(idxs)
                        sample_file = os.path.join(self.cache_dir, f"{sample_idx}.pkl")
                        with open(sample_file, 'rb') as f:
                            import pickle
                            sample_meta = pickle.load(f)
                        sample_features = sample_meta.get('features', {})
                        # Accept cache when either formula-only features OR subset features exist.
                        has_formula = sample_features.get('formulae_features', None) is not None
                        has_subset = sample_features.get('atom_subsets', None) is not None
                        if not (has_formula or has_subset):
                            self.cache_dir = None
                            self.df = pd.read_parquet(filename)
                        else:
                            self.df = pd.DataFrame({
                                'rdmol': [None] * nrows,
                                'spect': [None] * nrows,
                            })
                    except Exception:
                        # any error inspecting the cache -> fall back to parquet
                        self.cache_dir = None
                        self.df = pd.read_parquet(filename)
            except Exception:
                # if anything goes wrong reading the cache, fall back to parquet
                self.cache_dir = None
                self.df = pd.read_parquet(filename)
        else:
            # no usable cache found — read parquet (may raise if engine missing)
            self.cache_dir = None
            self.df = pd.read_parquet(filename)

        # Optional row-level filters for on-the-fly parquet mode only.
        # Keep cache-backed loading untouched so existing cache indices remain valid.
        if self.cache_dir is None:
            allowed_adducts = {x.strip().lower() for x in _parse_csv_env(os.environ.get('ALLOWED_ADDUCTS', '')) if x.strip()}
            allowed_instruments = {x.strip().lower() for x in _parse_csv_env(os.environ.get('ALLOWED_INSTRUMENTS', '')) if x.strip()}
            max_precursor_mz_raw = os.environ.get('MAX_PRECURSOR_MZ', '').strip()
            try:
                max_precursor_mz = float(max_precursor_mz_raw) if max_precursor_mz_raw else None
            except Exception:
                max_precursor_mz = None

            if allowed_adducts and 'adduct' in self.df.columns:
                adduct_series = self.df['adduct'].astype(str).str.strip().str.lower()
                self.df = self.df[adduct_series.isin(allowed_adducts)].reset_index(drop=True)

            if allowed_instruments and 'instrument_type' in self.df.columns:
                instrument_series = self.df['instrument_type'].astype(str).str.strip().str.lower()
                self.df = self.df[instrument_series.isin(allowed_instruments)].reset_index(drop=True)

            if max_precursor_mz is not None and 'precursor_mz' in self.df.columns:
                precursor_series = pd.to_numeric(self.df['precursor_mz'], errors='coerce')
                self.df = self.df[precursor_series.notna() & (precursor_series <= max_precursor_mz)].reset_index(drop=True)

        required_cols = ['rdmol', 'spect']
        for col in required_cols:
            assert col in self.df.columns, f'{col} must be in df'

        self.featurizer = create_mol_featurizer(spect_bin_config, featurizer_config)
        self.pred_featurizer = create_pred_featurizer(spect_bin_config, pred_featurizer_config)

        # Data policy switches (keep previous behavior as defaults).
        self.allow_heavy_atoms = os.environ.get('ALLOW_HEAVY_ATOMS', '0') == '1'
        try:
            self.max_mol_atoms = int(os.environ.get('MAX_MOL_ATOMS', '64'))
        except Exception:
            self.max_mol_atoms = 64
        self.impute_ce = os.environ.get('IMPUTE_CE', '1') == '1'
        try:
            self.default_ce = float(os.environ.get('DEFAULT_CE', '0'))
        except Exception:
            self.default_ce = 0.0
        self.default_adduct = os.environ.get('DEFAULT_ADDUCT', '[M+H]+').strip() or '[M+H]+'
        self.default_instrument = os.environ.get('DEFAULT_INSTRUMENT', 'unknown').strip() or 'unknown'
        self.default_ms_level = int(os.environ.get('DEFAULT_MS_LEVEL', '2'))
        self.impute_precursor = os.environ.get('IMPUTE_PRECURSOR', '1') == '1'
        self.impute_adduct = os.environ.get('IMPUTE_ADDUCT', '1') == '1'
        self.impute_instrument = os.environ.get('IMPUTE_INSTRUMENT', '1') == '1'
        self.impute_ms_level = os.environ.get('IMPUTE_MS_LEVEL', '1') == '1'

    # Function overview: _is_missing_value handles a specific workflow step in this module.
    @staticmethod
    def _is_missing_value(v):
        if v is None:
            return True
        if isinstance(v, str) and v.strip() == '':
            return True
        try:
            return bool(np.isnan(v))
        except Exception:
            return False

    # Function overview: _normalize_ce handles a specific workflow step in this module.
    def _normalize_ce(self, v):
        if self._is_missing_value(v):
            return float(self.default_ce) if self.impute_ce else v
        try:
            return float(v)
        except Exception:
            return float(self.default_ce) if self.impute_ce else v

    # Function overview: _normalize_adduct handles a specific workflow step in this module.
    def _normalize_adduct(self, v):
        if self._is_missing_value(v):
            return self.default_adduct if self.impute_adduct else v
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            if self._is_missing_value(v):
                return self.default_adduct if self.impute_adduct else v
            return int(v)
        return str(v)

    # Function overview: _normalize_instrument handles a specific workflow step in this module.
    def _normalize_instrument(self, v):
        if self._is_missing_value(v):
            return self.default_instrument if self.impute_instrument else v
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            if self._is_missing_value(v):
                return self.default_instrument if self.impute_instrument else v
            return int(v)
        return str(v)

    # Function overview: _normalize_precursor handles a specific workflow step in this module.
    def _normalize_precursor(self, v, fallback=None):
        if self._is_missing_value(v):
            if self.impute_precursor:
                if not self._is_missing_value(fallback):
                    try:
                        return float(fallback)
                    except Exception:
                        return 0.0
                return 0.0
            return v
        try:
            return float(v)
        except Exception:
            return 0.0 if self.impute_precursor else v

    # Function overview: _normalize_ms_level handles a specific workflow step in this module.
    def _normalize_ms_level(self, v):
        if self._is_missing_value(v):
            return int(self.default_ms_level) if self.impute_ms_level else v
        try:
            return int(v)
        except Exception:
            return int(self.default_ms_level) if self.impute_ms_level else v
    
    # Function overview: __len__ handles a specific workflow step in this module.
    def __len__(self):
        return len(self.df)
    
    # Load one sample with robust fallback order:
    # stage-1 cache load -> stage-2 policy filter -> stage-3 on-the-fly featurization.
    # If one row fails, it advances to the next row until a valid sample is found.
    def __getitem__(self, index):
        # 为避免因跳过样本而进入无限循环，限制尝试次数为数据集长度
        n = len(self.df)
        attempts = 0
        cur = int(index) % n
        while attempts < n:
            row = self.df.iloc[cur]
            # Stage 1: fastest path, load precomputed feature shard when available.
            # If a cache dir exists and contains this row, load it first
            if self.cache_dir is not None:
                cache_file = os.path.join(self.cache_dir, f"{cur}.pkl")
                err_file = os.path.join(self.cache_dir, f"{cur}.err")
                if os.path.exists(cache_file):
                    import pickle
                    try:
                        with open(cache_file, 'rb') as f:
                            meta = pickle.load(f)
                    except (EOFError, pickle.UnpicklingError, OSError, ValueError):
                        try:
                            os.remove(cache_file)
                        except Exception:
                            pass
                        self.cache_dir = None
                        self.df = pd.read_parquet(self.filename)
                        continue
                    features_dict = meta.get('features', {})
                    spect_cached = meta.get('spect_dense', None)
                    if spect_cached is None:
                        spect_cached = meta.get('spect', None)
                    spect_raw = meta.get('spect', None)
                    raw_ce = meta.get('collision_energy', None)
                    raw_adduct = meta.get('adduct', None)
                    raw_instrument = meta.get('instrument_type', None)
                    raw_precursor_mz = meta.get('precursor_mz', None)
                    raw_ms_level = meta.get('ms_level', None)
                    fallback_precursor_mz = meta.get('precursor_mz_fallback', None)
                    ce_missing = int(self._is_missing_value(raw_ce))
                    adduct_missing = int(self._is_missing_value(raw_adduct))
                    instrument_missing = int(self._is_missing_value(raw_instrument))
                    precursor_missing = int(self._is_missing_value(raw_precursor_mz))
                    ms_level_missing = int(self._is_missing_value(raw_ms_level))
                    res = {
                        'spect': spect_cached,
                        'spect_raw': spect_raw if spect_raw is not None else spect_cached,
                        'mol_id': meta.get('mol_id', row.get('mol_id', cur)),
                        'ce': self._normalize_ce(raw_ce),
                        'adduct': self._normalize_adduct(raw_adduct),
                        'instrument_type': self._normalize_instrument(raw_instrument),
                        'precursor_mz': self._normalize_precursor(raw_precursor_mz, fallback_precursor_mz),
                        'ms_level': self._normalize_ms_level(raw_ms_level),
                        'ce_missing': ce_missing,
                        'adduct_missing': adduct_missing,
                        'instrument_missing': instrument_missing,
                        'precursor_mz_missing': precursor_missing,
                        'ms_level_missing': ms_level_missing,
                        'ce_raw': raw_ce,
                        'adduct_raw': raw_adduct,
                        'instrument_raw': raw_instrument,
                        'precursor_mz_raw': raw_precursor_mz,
                        'ms_level_raw': raw_ms_level,
                        'condition_source': meta.get('condition_source', 'cache'),
                        **features_dict
                    }
                    res['input_idx'] = cur
                    return res
                # row known to fail during pre-cache; skip directly
                if os.path.exists(err_file):
                    attempts += 1
                    cur = (cur + 1) % n
                    continue

            # Stage 2: skip placeholder/non-materialized rows and policy-invalid molecules.
            # if placeholder dataframe (rdmol None) and no cache, skip
            if row.get('rdmol', None) is None:
                attempts += 1
                cur = (cur + 1) % n
                continue
            mol = Chem.Mol(row['rdmol'])
            # Optional heavy-element filtering (default keeps legacy behavior).
            atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            if (not self.allow_heavy_atoms) and any(x >= 20 for x in atomic_nums):
                attempts += 1
                cur = (cur + 1) % n
                continue

            # Optional molecule-size filter (default keeps legacy threshold 64).
            if self.max_mol_atoms > 0 and mol.GetNumAtoms() > self.max_mol_atoms:
                attempts += 1
                cur = (cur + 1) % n
                continue

            try:
                # Stage 3: row featurization path (or second cache check), then metadata normalization.
                # Output keys from this stage are consumed directly by train_ms_subsetnet.py
                # in prepare_batch_cpu() for tensorization and model input assembly.
                # If a cache dir exists and contains this row, load it instead
                if self.cache_dir is not None:
                    cache_file = os.path.join(self.cache_dir, f"{cur}.pkl")
                    err_file = os.path.join(self.cache_dir, f"{cur}.err")
                    if os.path.exists(cache_file):
                        import pickle
                        try:
                            with open(cache_file, 'rb') as f:
                                meta = pickle.load(f)
                        except (EOFError, pickle.UnpicklingError, OSError, ValueError):
                            try:
                                os.remove(cache_file)
                            except Exception:
                                pass
                            self.cache_dir = None
                            self.df = pd.read_parquet(self.filename)
                            continue
                        # meta contains 'features' (dict) and 'spect' etc.
                        features_dict = meta.get('features', {})
                        spect_cached = meta.get('spect_dense', None)
                        if spect_cached is None:
                            spect_cached = meta.get('spect', row['spect'])
                        spect_raw = meta.get('spect', row.get('spect', None))
                        raw_ce = meta.get('collision_energy', row.get('collision_energy', None))
                        raw_adduct = meta.get('adduct', row.get('adduct', None))
                        raw_instrument = meta.get('instrument_type', row.get('instrument_type', None))
                        raw_precursor_mz = meta.get('precursor_mz', row.get('precursor_mz', None))
                        raw_ms_level = meta.get('ms_level', row.get('ms_level', None))
                        fallback_precursor_mz = meta.get('precursor_mz_fallback', None)
                        ce_missing = int(self._is_missing_value(raw_ce))
                        adduct_missing = int(self._is_missing_value(raw_adduct))
                        instrument_missing = int(self._is_missing_value(raw_instrument))
                        precursor_missing = int(self._is_missing_value(raw_precursor_mz))
                        ms_level_missing = int(self._is_missing_value(raw_ms_level))
                        res = {
                            'spect': spect_cached,
                            'spect_raw': spect_raw if spect_raw is not None else spect_cached,
                            'ce': self._normalize_ce(raw_ce),
                            'adduct': self._normalize_adduct(raw_adduct),
                            'instrument_type': self._normalize_instrument(raw_instrument),
                            'precursor_mz': self._normalize_precursor(raw_precursor_mz, fallback_precursor_mz),
                            'ms_level': self._normalize_ms_level(raw_ms_level),
                            'ce_missing': ce_missing,
                            'adduct_missing': adduct_missing,
                            'instrument_missing': instrument_missing,
                            'precursor_mz_missing': precursor_missing,
                            'ms_level_missing': ms_level_missing,
                            'ce_raw': raw_ce,
                            'adduct_raw': raw_adduct,
                            'instrument_raw': raw_instrument,
                            'precursor_mz_raw': raw_precursor_mz,
                            'ms_level_raw': raw_ms_level,
                            'condition_source': meta.get('condition_source', 'cache'),
                            **features_dict
                        }
                        res['input_idx'] = cur
                        return res
                    if os.path.exists(err_file):
                        attempts += 1
                        cur = (cur + 1) % n
                        continue

                features_dict = self.featurizer(
                    mol,
                    row['spect'],
                    precursor_mz=row.get('precursor_mz', None),
                    precursor_formula=row.get('precursor_formula', None),
                    adduct=row.get('adduct', None),
                )
                
                # Optional imputations for missing metadata fields.
                from rdkit.Chem import Descriptors
                calc_precursor_mz = row.get('precursor_mz', None)
                if self.impute_precursor and (not calc_precursor_mz or calc_precursor_mz == 0):
                    calc_precursor_mz = Descriptors.ExactMolWt(mol) + 1.0078  # Assuming [M+H]+
                
                calc_adduct = row.get('adduct', None)
                if self.impute_adduct and (not calc_adduct or calc_adduct == 0):
                    calc_adduct = '[M+H]+'

                raw_ce = row.get('collision_energy', None)
                raw_adduct = row.get('adduct', None)
                raw_instrument = row.get('instrument_type', None)
                raw_precursor = row.get('precursor_mz', None)
                raw_ms_level = row.get('ms_level', None)
                ce_missing = int(self._is_missing_value(raw_ce))
                adduct_missing = int(self._is_missing_value(raw_adduct))
                instrument_missing = int(self._is_missing_value(raw_instrument))
                precursor_missing = int(self._is_missing_value(raw_precursor))
                ms_level_missing = int(self._is_missing_value(raw_ms_level))
                
                res = {
                    'spect': row['spect'],
                    'spect_raw': row['spect'],
                    'mol_id': row.get('mol_id', cur),
                    'ce': self._normalize_ce(raw_ce),
                    'adduct': self._normalize_adduct(calc_adduct),
                    'instrument_type': self._normalize_instrument(raw_instrument),
                    'precursor_mz': self._normalize_precursor(calc_precursor_mz),
                    'ms_level': self._normalize_ms_level(raw_ms_level),
                    'ce_missing': ce_missing,
                    'adduct_missing': adduct_missing,
                    'instrument_missing': instrument_missing,
                    'precursor_mz_missing': precursor_missing,
                    'ms_level_missing': ms_level_missing,
                    'ce_raw': raw_ce,
                    'adduct_raw': raw_adduct,
                    'instrument_raw': raw_instrument,
                    'precursor_mz_raw': raw_precursor,
                    'ms_level_raw': raw_ms_level,
                    'condition_source': 'row_or_imputed',
                    **features_dict
                }
                # 返回实际成功处理的行索引，便于追溯
                res['input_idx'] = cur
                return res
            except Exception:
                attempts += 1
                cur = (cur + 1) % n
                continue

        # 如果循环结束仍未找到可用样本，抛出错误以便上层可处理或定位问题
        raise IndexError(f"ParquetDataset: no valid molecule found near index {index} after {n} attempts")

    # Lightweight in-memory wrapper around a list of molecules for direct featurization.
class WrapperDataset:
    """
    Simple dataset to process mols
    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(
        self,
        mols,
        spect_bin_config,
        featurizer_config,
    ):
        self.featurizer = create_mol_featurizer(spect_bin_config, featurizer_config)
        self.mols = mols
        
    # Function overview: __len__ handles a specific workflow step in this module.
    def __len__(self):
        return len(self.mols)
    
    # Function overview: __getitem__ handles a specific workflow step in this module.
    def __getitem__(self, idx):
        mol = self.mols[idx]
        features_dict = self.featurizer(mol)
        out_dict = {
            **features_dict,
        }
        out_dict['input_idx'] = idx
        return out_dict

# Filter DB records by split/policy and return selected molecule IDs.
def filter_db_records(dataset_config, cv_splitter):
    """
    Filter out db records, filtering trhough the records as necessary. 
    """

    db_filename = dataset_config['db_filename']
    phase = dataset_config.get('phase', 'train')
    
    assert os.path.exists(db_filename)
    engine = create_engine(f"sqlite+pysqlite:///{db_filename}", future=True)
    metadata_obj = MetaData()
    print("reading from", db_filename)

    molecules = Table("molecules", metadata_obj, autoload_with=engine)

    cv_fp_field = dataset_config.get('cv_fp_field', 'morgan4_crc32')
    
    sql_stmt = select([molecules.c.id, molecules.c[cv_fp_field]])

    if 'filter_max_n' in dataset_config:
        sql_stmt = sql_stmt.where(molecules.c.atom_n <= dataset_config['filter_max_n'])

    if 'filter_max_mass' in dataset_config:
        sql_stmt = sql_stmt.where(molecules.c.mol_wt <= dataset_config['filter_max_mass'])

    if 'filter_max_unique_formulae' in dataset_config:
        sql_stmt = sql_stmt.where(molecules.c.unique_formulae <= dataset_config['filter_max_unique_formulae'])

    target_records = []
    with engine.connect() as conn:
        for row in conn.execute(sql_stmt):
            if cv_splitter.get_phase(None, row[cv_fp_field]) == phase:
                target_records.append(row['id'])

    return target_records

# Construct DBDataset after applying split-based ID filtering.
def make_db_dataset(dataset_config,
                    spect_bin_config, 
                    featurizer_config,
                    pred_config,
                    cv_splitter):
    db_ids = filter_db_records(dataset_config, cv_splitter)
    db_filename = dataset_config['db_filename']
    db_dataset = DBDataset(db_filename, 'id',
                           db_ids,
                           spect_bin_config,
                           featurizer_config,
                           pred_config)
    return db_dataset

# Construct ParquetDataset from config path.
def load_pq_dataset(dataset_config,
                    spect_bin_config, 
                    featurizer_config,
                    pred_config,
                    ):
    db_filename = dataset_config['db_filename']
    assert '.parquet' in db_filename or '.pq' in db_filename

    return ParquetDataset(
        db_filename,
        spect_bin_config,
        featurizer_config,
        pred_config,
    )
