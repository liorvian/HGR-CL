"""Experimental fixed, feature-conditioned incidence encoder (not a novelty claim).

Directed closed incoming neighborhoods define groups. Messages return along
the transpose incidence, unlike the historical two-forward-pass HSE. gmean2
is the mandatory same-incidence control. Both use two node-group-node rounds.
"""
import torch
from torch import nn
import torch.nn.functional as F


class GroupTrialEncoder(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant not in ('gmean2', 'gattn2'):
            raise ValueError(variant)
        self.variant = variant
        self._key = None
        self._input = None
        self._topology = None
        self._output = None

    def __getstate__(self):
        # Task-specific tensors and graph handles are not part of the model.
        state = super().__getstate__().copy()
        for name in ('_key', '_input', '_topology', '_output'):
            state[name] = None
        return state

    def _apply(self, fn, recurse=True):
        self._key = self._input = self._topology = self._output = None
        return super()._apply(fn, recurse=recurse)

    @staticmethod
    def incidence(graph, device):
        src, dst = graph.edges()
        src, dst = src.to(device).long(), dst.to(device).long()
        n = graph.num_nodes()
        ids = torch.arange(n, device=device)
        # One (member, center) per membership, including exactly one self.
        codes = torch.unique(torch.cat((dst*n+src, ids*n+ids)), sorted=True)
        return codes.remainder(n), torch.div(codes, n, rounding_mode='floor')

    @staticmethod
    def aggregate(values, source, target, weights, n):
        matrix = torch.sparse_coo_tensor(torch.stack((target, source)), weights,
                                         (n, n), device=values.device).coalesce()
        sums = torch.sparse.mm(matrix, values)
        denom = values.new_zeros(n).index_add_(0, target, weights)
        return sums / denom.clamp_min(1e-12).unsqueeze(1)

    @staticmethod
    def agreement(left, right, source, target):
        # Bound edge-feature temporaries; never materialize all M x d pairs.
        left = F.normalize(left, dim=1, eps=1e-12)
        right = F.normalize(right, dim=1, eps=1e-12)
        chunk = max(1, 1_000_000 // max(left.shape[1], 1))
        weights = left.new_empty(source.numel())
        for start in range(0, source.numel(), chunk):
            end = start + chunk
            cosine = (left[source[start:end]] * right[target[start:end]]).sum(1)
            # Temperature fixed at 1 before experiments; no trained parameters.
            weights[start:end] = cosine.clamp(-1, 1).exp()
        return weights

    @torch.no_grad()
    def forward(self, graph, features):
        if features.requires_grad:
            raise ValueError('Group trial is a fixed encoder for raw, non-trainable inputs')
        key = (features.data_ptr(), features._version, tuple(features.shape),
               tuple(features.stride()), features.device, features.dtype)
        if (self._topology is graph._graph and self._input is features
                and self._key == key):
            return self._output
        member, center = self.incidence(graph, features.device)
        n = graph.num_nodes()
        x = features
        unit = x.new_ones(member.numel())
        for _ in range(2):
            mean = self.aggregate(x, member, center, unit, n)
            if self.variant == 'gattn2':
                weights = self.agreement(x, mean, member, center)
                groups = self.aggregate(x, member, center, weights, n)
                back_weights = self.agreement(groups, x, center, member)
            else:
                groups, back_weights = mean, unit
            x = self.aggregate(groups, center, member, back_weights, n)
        self._key, self._input, self._topology = key, features, graph._graph
        self._output = x.detach()
        return self._output
