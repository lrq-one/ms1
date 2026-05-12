import numpy as np
import torch


class PeakFeatureMixin:
    def _build_peak_features(
        self,
        formulae_peaks_mass_idx,
        formulae_peaks_intensity,
        precursor_mz,
        batch_n,
        formula_n,
        peak_n,
        device,
        official_bin_width,
    ):
        mz = (formulae_peaks_mass_idx.float() + 0.5) * float(official_bin_width)
        inten = formulae_peaks_intensity.float()

        if precursor_mz is None:
            precursor = torch.ones((batch_n, 1, 1), dtype=torch.float32, device=device)
        else:
            precursor = precursor_mz
            if not torch.is_tensor(precursor):
                precursor = torch.as_tensor(precursor)
            precursor = precursor.to(device=device, dtype=torch.float32).reshape(-1)
            if precursor.shape[0] < batch_n:
                pad = torch.ones((batch_n - precursor.shape[0],), dtype=torch.float32, device=device)
                precursor = torch.cat([precursor, pad], dim=0)
            precursor = precursor[:batch_n].view(batch_n, 1, 1)

        mz_rel = mz / precursor.clamp_min(1e-6)
        gap_rel = (precursor - mz).clamp_min(0.0) / precursor.clamp_min(1e-6)
        inten_log = torch.log1p(inten.clamp_min(0.0))
        inten_raw = inten
        mz_raw = mz / 1500.0
        valid = ((formulae_peaks_mass_idx >= 0) & torch.isfinite(inten) & (inten > 0)).float()

        peak_feat = torch.stack(
            [
                mz_raw,
                mz_rel,
                gap_rel,
                inten_raw,
                inten_log,
                valid,
            ],
            dim=-1,
        )
        return peak_feat

    def _zero_precursor_bin_dense(self, dense_spect, precursor_mz, bin_width):
        if (not torch.is_tensor(dense_spect)) or precursor_mz is None:
            return dense_spect

        out = dense_spect.clone()
        if not torch.is_tensor(precursor_mz):
            precursor_mz = torch.as_tensor(precursor_mz)

        pmz = precursor_mz.to(device=out.device, dtype=torch.float32).reshape(-1)
        if pmz.shape[0] < int(out.shape[0]):
            pad = torch.zeros((int(out.shape[0]) - int(pmz.shape[0]),), dtype=torch.float32, device=out.device)
            pmz = torch.cat([pmz, pad], dim=0)
        pmz = pmz[: int(out.shape[0])]

        bin_idx = torch.floor(pmz / float(bin_width) + 1e-8).long()
        valid = torch.isfinite(pmz) & (bin_idx >= 0) & (bin_idx < int(out.shape[1]))
        if bool(valid.any().item()):
            row_idx = torch.arange(int(out.shape[0]), device=out.device)
            out[row_idx[valid], bin_idx[valid]] = 0.0
        return out

    def _to_2d_float(self, x, batch_n, device):
        if x is None:
            return torch.zeros((batch_n, 1), dtype=torch.float32, device=device)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.to(device)
        if x.dim() == 0:
            x = x.view(1, 1)
        elif x.dim() == 1:
            x = x.unsqueeze(-1)
        else:
            x = x.view(x.shape[0], -1)[:, :1]
        if x.shape[0] != batch_n:
            flat = x.reshape(-1)
            if flat.shape[0] < batch_n:
                pad_n = batch_n - flat.shape[0]
                flat = torch.cat([flat, torch.zeros((pad_n,), dtype=flat.dtype, device=device)], dim=0)
            x = flat[:batch_n].view(batch_n, 1)
        return x.float()

    def _coerce_frag_aux(self, x, batch_n, formula_n, device):
        if x is None or self.fragment_local_aux_dim <= 0:
            return None

        if not torch.is_tensor(x):
            try:
                x = torch.as_tensor(x)
            except Exception:
                return None

        x = x.to(device=device, dtype=torch.float32)

        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() > 3:
            x = x.reshape(x.shape[0], x.shape[1], -1)

        if x.dim() != 3:
            return None

        if x.shape[0] < batch_n:
            pad = torch.zeros(
                (batch_n - x.shape[0], x.shape[1], x.shape[2]),
                dtype=x.dtype,
                device=device,
            )
            x = torch.cat([x, pad], dim=0)
        x = x[:batch_n]

        if x.shape[1] < formula_n:
            pad = torch.zeros(
                (batch_n, formula_n - x.shape[1], x.shape[2]),
                dtype=x.dtype,
                device=device,
            )
            x = torch.cat([x, pad], dim=1)
        x = x[:, :formula_n, :]

        if x.shape[2] < self.fragment_local_aux_dim:
            pad = torch.zeros(
                (batch_n, formula_n, self.fragment_local_aux_dim - x.shape[2]),
                dtype=x.dtype,
                device=device,
            )
            x = torch.cat([x, pad], dim=-1)
        x = x[:, :, :self.fragment_local_aux_dim]

        return x

    def _gather_candidates_3d(self, x, idx):
        if (not torch.is_tensor(x)) or (not torch.is_tensor(idx)):
            return None
        B, K = idx.shape
        D = x.shape[-1]
        gather_idx = idx.unsqueeze(-1).expand(B, K, D)
        return torch.gather(x, 1, gather_idx)

    def _to_vocab_index(self, x, batch_n, device, vocab_size):
        if x is None:
            return torch.zeros((batch_n,), dtype=torch.long, device=device)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.to(device)
        if x.dim() == 0:
            x = x.view(1)
        elif x.dim() > 1:
            x = torch.argmax(x, dim=-1)
        x = x.long().reshape(-1)
        if x.shape[0] < batch_n:
            x = torch.cat([x, torch.zeros((batch_n - x.shape[0],), dtype=torch.long, device=device)], dim=0)
        x = x[:batch_n]
        return x % max(1, int(vocab_size))

    def _to_formulae_mask(self, x, batch_n, formula_n, device):
        if x is None:
            return torch.ones((batch_n, formula_n), dtype=torch.float32, device=device)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.to(device=device, dtype=torch.float32)
        target = batch_n * formula_n
        flat = x.reshape(-1)
        if flat.shape[0] < target:
            flat = torch.cat([flat, torch.ones((target - flat.shape[0],), dtype=torch.float32, device=device)], dim=0)
        mask = flat[:target].view(batch_n, formula_n)
        mask = (mask > 0.5).float()
        has_valid = (mask.sum(dim=1, keepdim=True) > 0)
        if not torch.all(has_valid):
            mask = torch.where(has_valid, mask, torch.ones_like(mask))
        return mask

    def _coerce_peak_tensor(self, x, batch_n, formula_n, device, dtype, fill_value):
        """
        Convert list / numpy / tensor -> [B, M, K] tensor.
        Return None if conversion fails.
        """
        if x is None:
            return None

        try:
            if torch.is_tensor(x):
                t = x.to(device=device, dtype=dtype)
            else:
                arr = np.asarray(x)
                t = torch.as_tensor(arr, device=device, dtype=dtype)
        except Exception:
            return None

        if t.dim() == 1:
            return None
        elif t.dim() == 2:
            # Avoid unintended broadcasting when a 2D peak tensor arrives.
            print(f"[WARNING] _coerce_peak_tensor received 2D input ({t.shape}), filling zeros.")
            t = torch.zeros(batch_n, formula_n, t.shape[-1], dtype=dtype, device=device)
        elif t.dim() > 3:
            t = t.reshape(t.shape[0], t.shape[1], -1)

        if t.dim() != 3:
            return None

        cur_b = int(t.shape[0])
        cur_m = int(t.shape[1])
        cur_k = int(t.shape[2])

        if cur_b < batch_n:
            pad = torch.full(
                (batch_n - cur_b, cur_m, cur_k),
                fill_value=fill_value,
                dtype=t.dtype,
                device=device,
            )
            t = torch.cat([t, pad], dim=0)
        t = t[:batch_n]

        if int(t.shape[1]) < formula_n:
            pad = torch.full(
                (batch_n, formula_n - int(t.shape[1]), int(t.shape[2])),
                fill_value=fill_value,
                dtype=t.dtype,
                device=device,
            )
            t = torch.cat([t, pad], dim=1)
        t = t[:, :formula_n, :]

        return t

    def _coerce_exact_peaks_tensor(self, x, batch_n, formula_n, device):
        """
        Convert list / numpy / tensor -> [B, M, K, 2] float tensor.
        Return None if conversion fails.
        """
        if x is None:
            return None

        try:
            if torch.is_tensor(x):
                t = x.to(device=device, dtype=torch.float32)
            else:
                arr = np.asarray(x)
                t = torch.as_tensor(arr, device=device, dtype=torch.float32)
        except Exception:
            return None

        if t.dim() == 3 and int(t.shape[-1]) >= 2:
            t = t.unsqueeze(0)
        elif t.dim() > 4:
            try:
                t = t.reshape(t.shape[0], t.shape[1], -1, t.shape[-1])
            except Exception:
                return None

        if t.dim() != 4 or int(t.shape[-1]) < 2:
            return None

        cur_b = int(t.shape[0])
        cur_m = int(t.shape[1])
        cur_k = int(t.shape[2])

        if cur_b < batch_n:
            pad = torch.zeros(
                (batch_n - cur_b, cur_m, cur_k, int(t.shape[-1])),
                dtype=t.dtype,
                device=device,
            )
            t = torch.cat([t, pad], dim=0)
        t = t[:batch_n]

        if int(t.shape[1]) < formula_n:
            pad = torch.zeros(
                (batch_n, formula_n - int(t.shape[1]), int(t.shape[2]), int(t.shape[3])),
                dtype=t.dtype,
                device=device,
            )
            t = torch.cat([t, pad], dim=1)
        t = t[:, :formula_n, :, :2]

        return t

    def _build_candidate_only_peak_summary(
        self,
        batch_n,
        formula_n,
        device,
        formulae_mask,
        off_idx,
        off_int,
        precursor_mz,
        official_bin_width,
        official_bin_n,
    ):
        feat_dim = 8
        out = torch.zeros((batch_n, formula_n, feat_dim), dtype=torch.float32, device=device)

        if (off_idx is None) or (off_int is None):
            return out
        if (not torch.is_tensor(off_idx)) or (not torch.is_tensor(off_int)):
            return out

        off_idx = off_idx.to(device=device, dtype=torch.long)
        off_int = off_int.to(device=device, dtype=torch.float32)

        if off_idx.dim() == 2:
            off_idx = off_idx.unsqueeze(0)
        if off_int.dim() == 2:
            off_int = off_int.unsqueeze(0)

        use_b = min(batch_n, int(off_idx.shape[0]), int(off_int.shape[0]))
        use_m = min(formula_n, int(off_idx.shape[1]), int(off_int.shape[1]))
        use_k = min(int(off_idx.shape[2]), int(off_int.shape[2]))
        if use_b <= 0 or use_m <= 0 or use_k <= 0:
            return out

        off_idx = off_idx[:use_b, :use_m, :use_k]
        off_int = off_int[:use_b, :use_m, :use_k]

        valid = (
            (off_idx >= 0)
            & (off_idx < int(official_bin_n))
            & torch.isfinite(off_int)
            & (off_int > 0)
        )

        cand_peak_n = valid.float().sum(dim=-1)
        cand_int_sum = (off_int * valid.float()).sum(dim=-1)
        cand_int_max = torch.where(valid, off_int, torch.zeros_like(off_int)).max(dim=-1).values

        cand_prob = off_int * valid.float()
        cand_prob = cand_prob / cand_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        cand_entropy = -(cand_prob * torch.log(cand_prob.clamp_min(1e-8))).sum(dim=-1)

        mz = (off_idx.float() + 0.5) * float(official_bin_width)
        mz = torch.where(valid, mz, torch.zeros_like(mz))

        mz_mean = mz.sum(dim=-1) / cand_peak_n.clamp_min(1.0)

        mz_centered = torch.where(valid, mz - mz_mean.unsqueeze(-1), torch.zeros_like(mz))
        mz_std = torch.sqrt((mz_centered ** 2).sum(dim=-1) / cand_peak_n.clamp_min(1.0))

        if precursor_mz is None:
            precursor = torch.zeros((use_b, 1), dtype=torch.float32, device=device)
        else:
            precursor = precursor_mz
            if not torch.is_tensor(precursor):
                precursor = torch.as_tensor(precursor)
            precursor = precursor.to(device=device, dtype=torch.float32).reshape(-1)
            if precursor.shape[0] < use_b:
                pad = torch.zeros((use_b - precursor.shape[0],), dtype=torch.float32, device=device)
                precursor = torch.cat([precursor, pad], dim=0)
            precursor = precursor[:use_b].view(use_b, 1)

        rel_gap = (precursor - mz_mean).clamp_min(0.0) / precursor.clamp_min(1e-6)
        rel_center = mz_mean / precursor.clamp_min(1e-6)

        top1_peak = torch.max(cand_prob, dim=-1).values

        feat = torch.stack(
            [
                cand_peak_n / 32.0,
                cand_int_sum,
                cand_int_max,
                cand_entropy / np.log(max(2, use_k)),
                mz_mean / 1500.0,
                mz_std / 300.0,
                rel_center.squeeze(-1),
                rel_gap.squeeze(-1),
            ],
            dim=-1,
        )

        if torch.is_tensor(formulae_mask):
            feat = feat * formulae_mask[:use_b, :use_m].unsqueeze(-1)

        out[:use_b, :use_m] = feat
        return out

    def _build_subset_peak_summary(
        self,
        subset_peaks_mass_idx,
        subset_peaks_intensity,
        precursor_mz,
        batch_n,
        subset_n,
        device,
        official_bin_width,
    ):
        feat_dim = 8
        out = torch.zeros((batch_n, subset_n, feat_dim), dtype=torch.float32, device=device)

        subset_peaks_mass_idx = self._coerce_peak_tensor(
            subset_peaks_mass_idx,
            batch_n=batch_n,
            formula_n=subset_n,
            device=device,
            dtype=torch.long,
            fill_value=-1,
        )
        subset_peaks_intensity = self._coerce_peak_tensor(
            subset_peaks_intensity,
            batch_n=batch_n,
            formula_n=subset_n,
            device=device,
            dtype=torch.float32,
            fill_value=0.0,
        )

        if subset_peaks_mass_idx is None or subset_peaks_intensity is None:
            return out

        use_b = min(batch_n, int(subset_peaks_mass_idx.shape[0]), int(subset_peaks_intensity.shape[0]))
        use_s = min(subset_n, int(subset_peaks_mass_idx.shape[1]), int(subset_peaks_intensity.shape[1]))
        use_k = min(int(subset_peaks_mass_idx.shape[2]), int(subset_peaks_intensity.shape[2]))
        if use_b <= 0 or use_s <= 0 or use_k <= 0:
            return out

        subset_peaks_mass_idx = subset_peaks_mass_idx[:use_b, :use_s, :use_k]
        subset_peaks_intensity = subset_peaks_intensity[:use_b, :use_s, :use_k]

        valid = (
            (subset_peaks_mass_idx >= 0)
            & torch.isfinite(subset_peaks_intensity)
            & (subset_peaks_intensity > 0)
        )

        peak_n = valid.float().sum(dim=-1)
        int_sum = (subset_peaks_intensity * valid.float()).sum(dim=-1)
        int_max = torch.where(valid, subset_peaks_intensity, torch.zeros_like(subset_peaks_intensity)).max(dim=-1).values

        prob = subset_peaks_intensity * valid.float()
        prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        entropy = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=-1)

        mz = (subset_peaks_mass_idx.float() + 0.5) * float(official_bin_width)
        mz = torch.where(valid, mz, torch.zeros_like(mz))
        mz_mean = mz.sum(dim=-1) / peak_n.clamp_min(1.0)

        mz_centered = torch.where(valid, mz - mz_mean.unsqueeze(-1), torch.zeros_like(mz))
        mz_std = torch.sqrt((mz_centered ** 2).sum(dim=-1) / peak_n.clamp_min(1.0))

        if precursor_mz is None:
            precursor = torch.zeros((use_b, 1), dtype=torch.float32, device=device)
        else:
            precursor = precursor_mz
            if not torch.is_tensor(precursor):
                precursor = torch.as_tensor(precursor)
            precursor = precursor.to(device=device, dtype=torch.float32).reshape(-1)
            if precursor.shape[0] < use_b:
                pad = torch.zeros((use_b - precursor.shape[0],), dtype=torch.float32, device=device)
                precursor = torch.cat([precursor, pad], dim=0)
            precursor = precursor[:use_b].view(use_b, 1)

        rel_gap = (precursor - mz_mean).clamp_min(0.0) / precursor.clamp_min(1e-6)
        rel_center = mz_mean / precursor.clamp_min(1e-6)

        feat = torch.stack(
            [
                peak_n / 32.0,
                int_sum,
                int_max,
                entropy / np.log(max(2, use_k)),
                mz_mean / 1500.0,
                mz_std / 300.0,
                rel_center.squeeze(-1),
                rel_gap.squeeze(-1),
            ],
            dim=-1,
        )
        out[:use_b, :use_s] = feat
        return out
