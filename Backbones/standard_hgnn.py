"""Standard HGNN propagation, independent of the historical HSE.

Reference: Feng et al., Hypergraph Neural Networks, AAAI 2019;
https://github.com/iMoonLab/HGNN (models/layers.py and utils/hypergraph_utils.py).
P = Dv^-1/2 H W De^-1 H.T Dv^-1/2; default hyperedge weights are one.
Graph-to-hypergraph construction is an explicit experimental choice, not part
of the HGNN layer definition: one complete symmetrized ego-hyperedge per node.
No truncation, learned hyperedge weights, or dense N x N propagation matrix.
"""
import torch
from torch import nn


def ego_incidence(graph, *, device, dtype):
    """Binary H[member, center]; union of incoming/outgoing neighbors + self."""
    n = graph.num_nodes()
    if n < 1:
        raise ValueError('HGNN requires a nonempty task graph')
    src, dst = (v.to(device=device, dtype=torch.long) for v in graph.edges())
    center = torch.arange(n, device=device)
    codes = torch.unique(torch.cat((src*n+dst, dst*n+src, center*n+center)))
    indices = torch.stack((codes.div(n, rounding_mode='floor'), codes.remainder(n)))
    return torch.sparse_coo_tensor(indices, torch.ones(codes.numel(), device=device,
                                   dtype=dtype), (n, n)).coalesce()


class HGNNPropagation:
    """Sparse exact factorization of the normalized HGNN propagation operator."""
    def __init__(self, incidence, edge_weights=None):
        if incidence.layout != torch.sparse_coo or incidence.ndim != 2:
            raise ValueError('incidence must be a 2-D sparse COO tensor')
        h = incidence.coalesce()
        if not h.is_floating_point() or not torch.isfinite(h.values()).all():
            raise ValueError('incidence must contain finite floating point values')
        if (h.values() < 0).any():
            raise ValueError('incidence must be nonnegative')
        n, m = h.shape
        w = (h.values().new_ones(m) if edge_weights is None else
             edge_weights.to(device=h.device, dtype=h.dtype))
        if w.shape != (m,) or not torch.isfinite(w).all() or (w <= 0).any():
            raise ValueError('hyperedge weights must be finite and positive, one per column')
        row, col = h.indices()
        de = h.values().new_zeros(m).index_add_(0, col, h.values())
        dv = h.values().new_zeros(n).index_add_(0, row, h.values()*w[col])
        if (de <= 0).any() or (dv <= 0).any():
            raise ValueError('empty hyperedges or uncovered nodes are not supported')
        self.h = h
        self.ht = h.transpose(0, 1).coalesce()
        self.node_scale = dv.rsqrt().unsqueeze(1)
        self.edge_scale = (w/de).unsqueeze(1)

    def __call__(self, x):
        if x.ndim != 2 or x.shape[0] != self.h.shape[0]:
            raise ValueError('feature rows must match incidence nodes')
        z = torch.sparse.mm(self.ht, x*self.node_scale) * self.edge_scale
        return torch.sparse.mm(self.h, z) * self.node_scale


class EgoHGNNBase(nn.Module):
    """Cache structure only; invalidate on DGL topology replacement or .to()."""
    def __init__(self):
        super().__init__()
        self.clear_cache()

    def clear_cache(self):
        self._topology = None
        self._structure_key = None
        self._propagation = None

    def propagation(self, graph, x):
        key = (graph.num_nodes(), graph.num_edges(), x.device, x.dtype)
        if self._topology is not graph._graph or self._structure_key != key:
            h = ego_incidence(graph, device=x.device, dtype=x.dtype)
            self._propagation = HGNNPropagation(h)
            self._topology = graph._graph
            self._structure_key = key
        return self._propagation

    def __getstate__(self):
        state = super().__getstate__().copy()
        for name in ('_topology', '_structure_key', '_propagation', '_input',
                     '_feature_key', '_output'):
            if name in state:
                state[name] = None
        return state

    def _apply(self, fn, recurse=True):
        self.clear_cache()
        return super()._apply(fn, recurse=recurse)


class FixedHGNNEncoder(EgoHGNNBase):
    """P^K X only: standard HGNN propagation, NOT the full learned HGNN model."""
    def __init__(self, k=2):
        super().__init__()
        if not isinstance(k, int) or k < 1:
            raise ValueError('k must be a positive integer')
        self.k = k

    def clear_cache(self):
        super().clear_cache()
        self._input = self._feature_key = self._output = None

    @torch.no_grad()
    def forward(self, graph, x):
        if x.requires_grad:
            raise ValueError('Fixed HGNN accepts fixed raw features only')
        key = (x.data_ptr(), x._version, tuple(x.shape), tuple(x.stride()), x.device, x.dtype)
        old_p = self._propagation
        p = self.propagation(graph, x)
        if p is old_p and self._input is x and self._feature_key == key:
            return self._output
        out = x
        for _ in range(self.k):
            out = p(out)
        self._input, self._feature_key, self._output = x, key, out.detach()
        return self._output

