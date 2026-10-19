
import dgl
import torch


def build_hyperedges_onehop(graph, max_hyperedge_size=None):
       
         
                    
                                                     
            
                                                 
       
    src, dst = graph.edges()
    src = src.tolist()
    dst = dst.tolist()

                               
    neighbors = {i: set() for i in range(graph.num_nodes())}
    for s, d in zip(src, dst):
        neighbors[d].add(s)

    hyperedges = []
    seen = set()
    for v in range(graph.num_nodes()):
                            
        edge_nodes = frozenset(neighbors[v] | {v})
        if edge_nodes in seen:
            continue
        seen.add(edge_nodes)
        if max_hyperedge_size is not None and len(edge_nodes) > max_hyperedge_size:
            continue
        hyperedges.append(edge_nodes)

    return hyperedges


def clique_expansion(hyperedges):
       
         
                                                                    
            
                                                            
       
    new_edges = set()
    for hedge in hyperedges:
        nodes = sorted(hedge)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                new_edges.add((u, v))
                new_edges.add((v, u))
    return new_edges


def build_expanded_graph(graph, mode='augment', hyperedge_construction='onehop', max_hyperedge_size=None):
       
         
                                        
                                              
                                                     
                                                              
            
                                           
                                 
       
    assert hyperedge_construction == 'onehop', \
        f"Unsupported hyperedge_construction: {hyperedge_construction}. Only 'onehop' is supported."
    assert mode in ('augment', 'replace'), \
        f"Unsupported mode: {mode}. Choose 'augment' or 'replace'."

    hyperedges = build_hyperedges_onehop(graph, max_hyperedge_size=max_hyperedge_size)
    new_edges = clique_expansion(hyperedges)

    if mode == 'augment':
        orig_src, orig_dst = graph.edges()
        orig_edge_set = set(zip(orig_src.tolist(), orig_dst.tolist()))
        combined_edges = orig_edge_set | new_edges
                                                  
        combined_edges = {(u, v) for u, v in combined_edges if u != v}
    else:  # replace
        combined_edges = {(u, v) for u, v in new_edges if u != v}

    if len(combined_edges) > 0:
        srcs, dsts = zip(*combined_edges)
        srcs = torch.tensor(srcs, dtype=torch.long)
        dsts = torch.tensor(dsts, dtype=torch.long)
    else:
        srcs = torch.tensor([], dtype=torch.long)
        dsts = torch.tensor([], dtype=torch.long)

    new_graph = dgl.graph((srcs, dsts), num_nodes=graph.num_nodes())

                
    for key, val in graph.ndata.items():
        new_graph.ndata[key] = val.clone()

                          
    new_graph = dgl.add_self_loop(new_graph)

    return new_graph, hyperedges


def compute_hypergraph_stats(orig_graph, hyperedges, expanded_graph):
       
         
                                              
                                     
                                                               
            
                   
       
    n_hyperedges = len(hyperedges)
    if n_hyperedges > 0:
        sizes = [len(h) for h in hyperedges]
        avg_size = sum(sizes) / n_hyperedges
        max_size = max(sizes)
    else:
        avg_size = 0.0
        max_size = 0

    stats = {
        'n_nodes': orig_graph.num_nodes(),
        'n_edges_orig': orig_graph.num_edges(),
        'n_edges_expanded': expanded_graph.num_edges(),
        'n_hyperedges': n_hyperedges,
        'avg_hyperedge_size': round(avg_size, 2),
        'max_hyperedge_size_actual': max_size,
    }
    return stats
