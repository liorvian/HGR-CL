"""Directed four-step controls for the HSE trial's preprocessing confounds."""
import torch
from training.hse_group_trial import GroupTrialEncoder


class GraphControlEncoder(GroupTrialEncoder):
    def __init__(self, variant):
        super().__init__('gmean2')
        if variant not in ('gclean4', 'grow4', 'gbi4'):
            raise ValueError(variant)
        self.variant = variant

    @torch.no_grad()
    def forward(self, graph, features):
        if features.requires_grad:
            raise ValueError('Fixed graph controls require non-trainable inputs')
        key = (features.data_ptr(), features._version, tuple(features.shape),
               tuple(features.stride()), features.device, features.dtype)
        if (self._topology is graph._graph and self._input is features
                and self._key == key):
            return self._output
        member, center = self.incidence(graph, features.device)
        n = graph.num_nodes()
        if self.variant == 'gbi4':
            # Union of both directions, deduplicated, with one self membership.
            codes = torch.unique(torch.cat((center*n+member, member*n+center)), sorted=True)
            member = codes.remainder(n)
            center = torch.div(codes, n, rounding_mode='floor')
        unit = features.new_ones(member.numel())
        degree = features.new_zeros(n).index_add_(0, center, unit)
        if self.variant == 'gclean4':
            # Same incoming-degree symmetric normalization as original graph4,
            # but deduplicated directed edges and exactly one self membership.
            norm = degree.clamp_min(1).rsqrt()
            weights = norm[member] * norm[center]
        else:
            # Row-normalized incoming mean; direction still forward four times.
            weights = degree[center].clamp_min(1).reciprocal()
        matrix = torch.sparse_coo_tensor(torch.stack((center, member)), weights,
                                         (n, n), device=features.device).coalesce()
        x = features
        for _ in range(4):
            x = torch.sparse.mm(matrix, x)
        self._key, self._input, self._topology = key, features, graph._graph
        self._output = x.detach()
        return self._output
