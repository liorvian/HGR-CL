import random
import numpy as np
from numpy.random import choice
import torch
import torch.nn as nn
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CoraGraphDataset, CoraFullDataset, register_data_args, RedditDataset


class random_select(nn.Module):
    def __init__(self,args=None):
        super(random_select, self).__init__()
    def forward(self, ids_per_cls_train, budget, neighbor_agg_model=None, graph=None, topo_vecs=None, max_ratio_per_cls = 1.0):
        store_ids = []
        for i,ids in enumerate(ids_per_cls_train):
            budget_ = min(budget, int(max_ratio_per_cls * len(ids))) if isinstance(budget, int) else int(budget * len(ids))
            store_ids.extend(random.sample(ids, budget_))
        return store_ids

class cover_max_select_01(nn.Module):
    # in this version, the nodes are simultaneously sampled according to their covered size, instead of repeatedly increasing single nodes and evaluating the covered areas
    def __init__(self,args=None):
        super().__init__()

    def forward(self, ids_per_cls, budget, neighbor_agg_model=None, graph=None, topo_vecs=None, max_ratio_per_cls = 1.0):
        n_nodes = graph.num_nodes()
        d = torch.ones(n_nodes)
        indicators = torch.diag(d).bool().cuda(graph.device)
        cover_per_node = neighbor_agg_model(graph, indicators).bool()
        return self.select(cover_per_node, budget, ids_per_cls, n_nodes)

    def select(self, cover_per_node, budget, ids_per_cls, n_nodes):
        ids_selected = []
        for ids in ids_per_cls:
            judge = cover_per_node.bool().sum(1)[ids]
            judge_prob = (judge/judge.sum()).cpu().double()
            judge_prob /= judge_prob.sum()
            ids_selec = choice(ids, min(budget, len(ids)),replace=False, p=judge_prob)
            ids_selected.extend(ids_selec)
        return ids_selected


class cover_max_select_02(nn.Module):
    # compared to 01, this version use degree as surrogate of the coverage
    def __init__(self,args=None):
        super().__init__()
        self.covered_nodes = []
        self.covered_nodes_random = []
        self.covered_nodes_herd = []
        self.random_select = random_select()

    def forward(self, ids_per_cls, budget, neighbor_agg_model=None, graph=None, topo_vecs=None, max_ratio_per_cls = 1.0, batch_size=10):
        n_nodes = graph.num_nodes()

        ind = graph.in_degrees()
        selected_ids = self.select(ind, budget, ids_per_cls, n_nodes)
        he_cov = compute_he_coverage(selected_ids, graph)
        # print(f"[HE-Cov] cover_max_select_02 | budget={budget} | selected={len(selected_ids)} | HE-Cov={he_cov*100:.2f}%")
        return selected_ids

    def select(self, cover_per_node, budget, ids_per_cls, n_nodes):
        ids_selected = []
        for ids in ids_per_cls:
            judge = cover_per_node[ids].half()
            ids_of_ids = torch.multinomial(judge, min(budget, len(ids)), replacement=False).tolist()
            ids_selec = torch.tensor(ids)[ids_of_ids].tolist()
            ids_selected.extend(ids_selec)
        return ids_selected


class hyperedge_aware_select(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        if args is not None and hasattr(args, 'tem_args') and isinstance(args.tem_args, dict):
            self.alpha = args.tem_args.get('he_alpha', 1.0)
            self.beta  = args.tem_args.get('he_beta',  1.0)
            self.gamma = args.tem_args.get('he_gamma', 1.0)
        else:
            self.alpha = 1.0
            self.beta  = 1.0
            self.gamma = 1.0

    def _build_neighbors(self, graph):
        n_nodes = graph.num_nodes()
        src, dst = graph.edges()
        src, dst = src.tolist(), dst.tolist()
        neighbors = {v: set() for v in range(n_nodes)}
        for s, d in zip(src, dst):
            neighbors[d].add(s)
            neighbors[s].add(d)
        return neighbors

    def _compute_scores(self, graph, ids_per_cls_train):
        neighbors = self._build_neighbors(graph)
        n_nodes = graph.num_nodes()

                                   
        task_nodes = set()
        for ids in ids_per_cls_train:
            task_nodes.update(ids)

        he_count        = torch.zeros(n_nodes, dtype=torch.float)
        avg_he_size     = torch.zeros(n_nodes, dtype=torch.float)
        unique_coverage = torch.zeros(n_nodes, dtype=torch.float)

        for v in range(n_nodes):
                                                              
                                                 
            anchors = neighbors[v] | {v}

            # he_count(v)
            he_count[v] = len(anchors)

                                                
            sizes = [len(neighbors[u]) + 1 for u in anchors]
            avg_he_size[v] = sum(sizes) / len(sizes)

                                                          
            covered = set()
            for u in anchors:
                covered.update((neighbors[u] | {u}) & task_nodes)
            unique_coverage[v] = len(covered)

        return he_count, avg_he_size, unique_coverage

    def forward(self, ids_per_cls_train, budget, neighbor_agg_model=None, graph=None,
                topo_vecs=None, max_ratio_per_cls=1.0):
        he_count, avg_he_size, unique_coverage = self._compute_scores(graph, ids_per_cls_train)

                               
        all_ids = [i for ids in ids_per_cls_train for i in ids]
        all_ids_t = torch.tensor(all_ids, dtype=torch.long)

        def minmax_norm(x):
            vals = x[all_ids_t]
            mn, mx = vals.min(), vals.max()
            if (mx - mn).item() < 1e-8:
                return torch.ones_like(x)              
            return (x - mn) / (mx - mn)

        score = (self.alpha * minmax_norm(he_count)
               + self.beta  * minmax_norm(avg_he_size)
               + self.gamma * minmax_norm(unique_coverage))


                                    
        ids_selected = []
        for ids in ids_per_cls_train:
            budget_ = (min(budget, int(max_ratio_per_cls * len(ids)))
                       if isinstance(budget, int) else int(budget * len(ids)))
            cls_scores = score[torch.tensor(ids, dtype=torch.long)].float().clamp(min=1e-8)
            sampled = torch.multinomial(cls_scores, min(budget_, len(ids)), replacement=False).tolist()
            ids_selected.extend([ids[i] for i in sampled])

        he_cov = compute_he_coverage(ids_selected, graph)
        # print(f"[HE-Cov] hyperedge_aware | budget={budget} | selected={len(ids_selected)} | HE-Cov={he_cov*100:.2f}%")

        return ids_selected


def compute_he_coverage(selected_ids, graph):
    n_nodes = graph.num_nodes()
    src, dst = graph.edges()
    src, dst = src.tolist(), dst.tolist()
    neighbors = {v: set() for v in range(n_nodes)}
    for s, d in zip(src, dst):
        neighbors[s].add(d)
        neighbors[d].add(s)
    covered = set()
    for v in selected_ids:
        covered.update(neighbors[v] | {v})
    return len(covered) / max(n_nodes, 1)
