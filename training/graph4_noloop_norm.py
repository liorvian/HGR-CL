"""Four normalized aggregations with ALL self-loops removed.

Output is (R A0 R)^4 X; R uses clamped loop-free in-degrees.
Original directions and parallel non-self edges are retained.
"""
import torch


class Graph4NoLoopNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clear_cache()

    def clear_cache(self):
        self._topology = self._input = self._key = self._output = None

    def __getstate__(self):
        state = super().__getstate__().copy()
        for key in ('_topology', '_input', '_key', '_output'):
            state[key] = None
        return state

    def _apply(self, fn, recurse=True):
        self.clear_cache()
        return super()._apply(fn, recurse=recurse)

    @torch.no_grad()
    def forward(self, graph, features):
        if features.requires_grad:
            raise ValueError('This fixed encoder requires non-trainable input features')
        if features.ndim != 2 or features.shape[0] != graph.num_nodes():
            raise ValueError('Feature rows must match graph nodes')
        key = (graph.num_nodes(), graph.num_edges(), features.data_ptr(),
               features._version, tuple(features.shape), tuple(features.stride()),
               features.device, features.dtype)
        if self._topology is graph._graph and self._input is features and self._key == key:
            return self._output
        src, dst = graph.edges()
        keep = src != dst
        src = src[keep].to(device=features.device, dtype=torch.long)
        dst = dst[keep].to(device=features.device, dtype=torch.long)
        n = graph.num_nodes()
        matrix = torch.sparse_coo_tensor(torch.stack((dst, src)),
            features.new_ones(src.numel()), (n,n)).coalesce()
        degree = torch.bincount(dst, minlength=n).to(features.dtype)
        norm = degree.clamp(min=1).pow(-0.5).unsqueeze(1)
        output = features
        for _ in range(4):
            output = torch.sparse.mm(matrix, output * norm) * norm
        if not torch.isfinite(output).all():
            raise FloatingPointError('Normalized loop-free Graph4 produced non-finite values. '
                                     'No automatic rescaling is applied.')
        self._topology, self._input = graph._graph, features
        self._key, self._output = key, output.detach()
        return self._output
