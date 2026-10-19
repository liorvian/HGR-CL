"""Isolated P1-A encoders. Original training/backbone files are unchanged."""
import time
import torch
import dgl
import dgl.function as fn
from Backbones.gnns import HypergraphSGC_Agg


class P1AEncoder(torch.nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.depth = {'raw':0, 'graph1':1, 'graph2':2, 'graph4':4, 'hse2':4}[variant]
        self.hse = HypergraphSGC_Agg(k=2) if variant == 'hse2' else None
        self._topology = None
        self._graph = None
        self.seconds = {'train':0., 'eval':0.}
        self.calls = {'train':0, 'eval':0}

    def forward(self, graph, features):
        # Preserve the input graph used by HMS/SRL. Only the encoder adds loops.
        if features.is_cuda: torch.cuda.synchronize(features.device)
        start = time.perf_counter()
        if self._topology is not graph._graph:
            self._topology = graph._graph
            if self.hse is not None:
                # Isolated experiment guard: never reuse topology by size alone.
                self.hse._cache_graph_id = None
            elif self.depth:
                self._graph = dgl.add_self_loop(graph)
                self._norm = self._graph.in_degrees().float().clamp(min=1).pow(-.5).unsqueeze(1)
        if self.hse is not None:
            output = self.hse(graph, features)
        elif self.depth == 0:
            output = features
        else:
            output = features
            with self._graph.local_scope():
                for _ in range(self.depth):
                    self._graph.ndata['h'] = output * self._norm
                    self._graph.update_all(fn.copy_u('h','m'),fn.sum('m','h'))
                    output = self._graph.ndata['h'] * self._norm
        if features.is_cuda: torch.cuda.synchronize(features.device)
        mode = 'train' if self.training else 'eval'
        self.seconds[mode] += time.perf_counter()-start
        self.calls[mode] += 1
        return output
