# rassp/model/int_embedder.py
import torch as th
import torch.nn as nn
import numpy as np

DEFAULT_MAX_COUNT_INT = 255
DEFAULT_NUM_EXTRA_EMBEDDINGS = 1


class IntFeaturizer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        max_count_int: int = DEFAULT_MAX_COUNT_INT,
        num_extra_embeddings: int = DEFAULT_NUM_EXTRA_EMBEDDINGS,
    ):
        super().__init__()
        self.max_count_int = int(max_count_int)
        self.num_extra_embeddings = int(num_extra_embeddings)
        self.embedding_dim = int(embedding_dim)

        weights = th.zeros(self.num_extra_embeddings, self.embedding_dim)
        self._extra_embeddings = nn.Parameter(weights, requires_grad=True)
        nn.init.normal_(self._extra_embeddings, 0.0, 1.0)

    def forward(self, tensor: th.Tensor) -> th.Tensor:
        orig_shape = tensor.shape
        tensor = tensor.long()

        extra_embed_mask = (tensor >= self.max_count_int).float()
        clamped_tensor = th.clamp(tensor, min=0)

        norm_idx = th.clamp(clamped_tensor, max=self.max_count_int - 1)
        norm_embeds = self.int_to_feat_matrix[norm_idx]

        extra_idx = th.maximum(
            clamped_tensor,
            self.max_count_int * th.ones_like(clamped_tensor),
        ) - self.max_count_int
        extra_idx = th.clamp(extra_idx, min=0, max=self.num_extra_embeddings - 1)
        extra_embeds = self._extra_embeddings[extra_idx]

        out_tensor = (
            (1.0 - extra_embed_mask).unsqueeze(-1) * norm_embeds
            + extra_embed_mask.unsqueeze(-1) * extra_embeds
        )

        # flatten the final two dims into the feature dim
        temp_out = out_tensor.reshape(*orig_shape[:-1], -1)
        return temp_out

    @property
    def num_dim(self) -> int:
        return int(self.int_to_feat_matrix.shape[1])


class FourierFeaturizer(IntFeaturizer):
    def __init__(
        self,
        max_count_int: int = DEFAULT_MAX_COUNT_INT,
        num_extra_embeddings: int = DEFAULT_NUM_EXTRA_EMBEDDINGS,
    ):
        num_freqs = int(np.ceil(np.log2(max_count_int))) + 2
        freqs = 0.5 ** th.arange(num_freqs, dtype=th.float32)
        freqs_time_2pi = 2 * np.pi * freqs

        super().__init__(
            embedding_dim=2 * freqs_time_2pi.shape[0],
            max_count_int=max_count_int,
            num_extra_embeddings=num_extra_embeddings,
        )

        combo = (
            th.arange(self.max_count_int, dtype=th.float32)[:, None]
            * freqs_time_2pi[None, :]
        )
        all_features = th.cat([th.cos(combo), th.sin(combo)], dim=1)
        self.int_to_feat_matrix = nn.Parameter(all_features.float(), requires_grad=False)


class FourierFeaturizerSines(IntFeaturizer):
    def __init__(
        self,
        max_count_int: int = DEFAULT_MAX_COUNT_INT,
        num_extra_embeddings: int = DEFAULT_NUM_EXTRA_EMBEDDINGS,
    ):
        num_freqs = int(np.ceil(np.log2(max_count_int))) + 2
        freqs = (0.5 ** th.arange(num_freqs, dtype=th.float32))[2:]
        freqs_time_2pi = 2 * np.pi * freqs

        super().__init__(
            embedding_dim=freqs_time_2pi.shape[0],
            max_count_int=max_count_int,
            num_extra_embeddings=num_extra_embeddings,
        )

        combo = (
            th.arange(self.max_count_int, dtype=th.float32)[:, None]
            * freqs_time_2pi[None, :]
        )
        self.int_to_feat_matrix = nn.Parameter(th.sin(combo).float(), requires_grad=False)


class FourierFeaturizerAbsoluteSines(IntFeaturizer):
    def __init__(
        self,
        max_count_int: int = DEFAULT_MAX_COUNT_INT,
        num_extra_embeddings: int = DEFAULT_NUM_EXTRA_EMBEDDINGS,
    ):
        num_freqs = int(np.ceil(np.log2(max_count_int))) + 2
        freqs = (0.5 ** th.arange(num_freqs, dtype=th.float32))[2:]
        freqs_time_2pi = 2 * np.pi * freqs

        super().__init__(
            embedding_dim=freqs_time_2pi.shape[0],
            max_count_int=max_count_int,
            num_extra_embeddings=num_extra_embeddings,
        )

        combo = (
            th.arange(self.max_count_int, dtype=th.float32)[:, None]
            * freqs_time_2pi[None, :]
        )
        self.int_to_feat_matrix = nn.Parameter(th.abs(th.sin(combo)).float(), requires_grad=False)


class RBFFeaturizer(IntFeaturizer):
    def __init__(
        self,
        num_funcs: int = 32,
        max_count_int: int = DEFAULT_MAX_COUNT_INT,
        num_extra_embeddings: int = DEFAULT_NUM_EXTRA_EMBEDDINGS,
    ):
        super().__init__(
            embedding_dim=num_funcs,
            max_count_int=max_count_int,
            num_extra_embeddings=num_extra_embeddings,
        )
        width = (self.max_count_int - 1) / max(1, num_funcs)
        centers = th.linspace(0, self.max_count_int - 1, num_funcs)

        pre_exp = (
            -0.5
            * ((th.arange(self.max_count_int)[:, None] - centers[None, :]) / width) ** 2
        )
        feats = th.exp(pre_exp)
        self.int_to_feat_matrix = nn.Parameter(feats.float(), requires_grad=False)


class OneHotFeaturizer(IntFeaturizer):
    def __init__(
        self,
        max_count_int: int = DEFAULT_MAX_COUNT_INT,
        num_extra_embeddings: int = DEFAULT_NUM_EXTRA_EMBEDDINGS,
    ):
        super().__init__(
            embedding_dim=max_count_int,
            max_count_int=max_count_int,
            num_extra_embeddings=num_extra_embeddings,
        )
        feats = th.eye(self.max_count_int)
        self.int_to_feat_matrix = nn.Parameter(feats.float(), requires_grad=False)


class LearnedFeaturizer(IntFeaturizer):
    def __init__(
        self,
        feature_dim: int = 32,
        max_count_int: int = DEFAULT_MAX_COUNT_INT,
        num_extra_embeddings: int = DEFAULT_NUM_EXTRA_EMBEDDINGS,
    ):
        super().__init__(
            embedding_dim=feature_dim,
            max_count_int=max_count_int,
            num_extra_embeddings=num_extra_embeddings,
        )
        weights = th.zeros(self.max_count_int, feature_dim)
        self.int_to_feat_matrix = nn.Parameter(weights, requires_grad=True)
        nn.init.normal_(self.int_to_feat_matrix, 0.0, 1.0)


def get_embedder(embedder: str, **kwargs):
    if embedder == "fourier":
        return FourierFeaturizer(**kwargs)
    if embedder == "rbf":
        return RBFFeaturizer(**kwargs)
    if embedder == "one-hot":
        return OneHotFeaturizer(**kwargs)
    if embedder == "learnt":
        return LearnedFeaturizer(**kwargs)
    if embedder == "fourier-sines":
        return FourierFeaturizerSines(**kwargs)
    if embedder == "abs-sines":
        return FourierFeaturizerAbsoluteSines(**kwargs)
    raise NotImplementedError(f"Unknown embedder: {embedder}")