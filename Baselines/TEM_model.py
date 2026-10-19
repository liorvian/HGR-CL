from .TEM_utils import *
import torch.nn.functional as F
from training.memory_audit import MemoryAudit
from training.p1e_routing import choose_edge

def _compute_avg_deg(g):
    src, _ = g.edges()
    num_nodes = g.num_nodes()
    num_edges = src.numel() if hasattr(src, "numel") else len(src)
    avg_deg = float(num_edges) / max(int(num_nodes), 1)
    # print(f"[DEG DEBUG] num_nodes={num_nodes}, num_edges={num_edges}, avg_deg={avg_deg:.6f}")
    return avg_deg, num_edges, num_nodes


sampler_types = {'random_select': random_select, 'cover_max_select_01':cover_max_select_01, 'cover_max_select_02':cover_max_select_02,
                 'hyperedge_aware': hyperedge_aware_select}


class NET(torch.nn.Module):
    def __init__(self,
                 model,
                 task_manager,
                 args):
        super(NET, self).__init__()

        self.task_manager = task_manager

        # setup network
        self.net = model

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup losses
        self.ce = torch.nn.functional.cross_entropy

        # setup memories
        self.current_task = -1
        self.sampler = sampler_types[args.tem_args['sampler']](args) if not isinstance(args.tem_args['sampler'], list) else sampler_types[args.tem_args['sampler'][0]](args)
        self.TEM_vecs = torch.tensor([]).cuda(args.gpu)
        self.TEM_labels = torch.tensor([]).long().cuda(args.gpu)
        budget_raw = args.tem_args['budget']
        self.budget = int(budget_raw) if (isinstance(budget_raw, float) and budget_raw >= 1.0) else budget_raw
        self._prev_task_cache = None
        self.memory_audit = MemoryAudit(args, self.budget)
        self._avg_deg = None

    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, args, g, features, labels_all, t, train_ids, ids_per_cls, dataset):
        self.net.train()
        labels = labels_all[train_ids]
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        self.opt.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        topo_vecs = self.net.neighbor_agg(g, features)
        input_feat_concat = torch.cat([topo_vecs[train_ids], self.TEM_vecs], dim=0)
        labels_concat = torch.cat([labels, self.TEM_labels], dim=0)

        if t != self.current_task:
            self.current_task = t
            torch.cuda.empty_cache()

            if self._avg_deg is None:
                self._avg_deg, num_edges, num_nodes = _compute_avg_deg(g)
                # print(f"[SC Router] dataset={args.dataset}, avg_deg={self._avg_deg:.6f}, num_edges={num_edges}, num_nodes={num_nodes}")

            if self._prev_task_cache is not None:
                prev = self._prev_task_cache
                selected_ids = self.sampler(prev['ids_per_cls_train'], self.budget,
                                            neighbor_agg_model=self.net.neighbor_agg,
                                            graph=prev['graph'], topo_vecs=prev['topo_vecs'])
                self.TEM_vecs = torch.cat([self.TEM_vecs, prev['topo_vecs'][selected_ids].detach()], dim=0)
                self.TEM_labels = torch.cat([self.TEM_labels, prev['labels_all'][selected_ids]], dim=0)

            self.memory_audit.snapshot('task_entry_after_memory_update', t, self)


        if args.cls_balance:
            n_per_cls = [(labels_concat == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        output, _ = self.net.feat_trans(g, input_feat_concat)

        if args.classifier_increase:
            loss = self.ce(output[:,offset1:offset2], labels_concat, weight=loss_w_[offset1: offset2])
        else:
            loss = self.ce(output, labels_concat, weight=loss_w_)

        lambda_sc = args.tem_args.get('lambda_sc', 0.0)
        p1e_mode = args.tem_args.get('p1e_mode', 'switch')
        if p1e_mode == 'none':
            lambda_sc = 0.0
        if lambda_sc > 0.0:
            deg_tau = float(args.tem_args.get('deg_tau', 3.0))
            # print(f"[SC Router] avg_deg={self._avg_deg:.6f}, deg_tau={deg_tau:.6f}")
            use_graph_edge_sc = choose_edge(p1e_mode, self._avg_deg, deg_tau)

            sc_margin = float(args.tem_args.get('sc_margin', 0.5))
            N_sc_base = int(args.tem_args.get('N_sc', 64))
            if use_graph_edge_sc:
                                        
                                                                       
                if len(train_ids) >= 4:
                    N_sc = min(N_sc_base, len(train_ids))
                    perm_sc = torch.randperm(len(train_ids))[:N_sc].tolist()
                    train_ids_list = train_ids if isinstance(train_ids, list) else train_ids.tolist()
                    sc_ids = [int(train_ids_list[i]) for i in perm_sc]
                    src_sc, dst_sc = g.edges()
                    src_sc, dst_sc = src_sc.tolist(), dst_sc.tolist()
                    sc_set = set(sc_ids)
                    sc_id_to_idx = {v: i for i, v in enumerate(sc_ids)}
                    neighbors_sc = {i: [] for i in range(N_sc)}
                    for s, d in zip(src_sc, dst_sc):
                        if s in sc_set and d in sc_set:
                            neighbors_sc[sc_id_to_idx[s]].append(sc_id_to_idx[d])
                            neighbors_sc[sc_id_to_idx[d]].append(sc_id_to_idx[s])
                    sc_tensor = torch.tensor(sc_ids, dtype=torch.long, device=topo_vecs.device)
                    h_sc = topo_vecs[sc_tensor]
                    k_top_nb = int(args.tem_args.get('k_top_nb', 0))
                    if k_top_nb > 0:
                        with torch.no_grad():
                            topo_nb_norm = F.normalize(h_sc, p=2, dim=1)
                            sim_nb = torch.mm(topo_nb_norm, topo_nb_norm.t())
                        for i in range(N_sc):
                            nbrs = neighbors_sc[i]
                            if len(nbrs) > k_top_nb:
                                neighbors_sc[i] = sorted(nbrs, key=lambda j: sim_nb[i, j].item(), reverse=True)[:k_top_nb]
                    for layer_i, layer in enumerate(self.net.feat_trans_layers[:-1]):
                        h_sc = layer(h_sc)
                        if self.net.bn:
                            h_sc = self.net.bns[layer_i](h_sc)
                        h_sc = F.relu(h_sc)
                        h_sc = F.dropout(h_sc, p=self.net.dropout, training=self.training)
                    h_sc_norm = F.normalize(h_sc, p=2, dim=1)
                    anc_list, pos_list, neg_list = [], [], []
                    for i in range(N_sc):
                        pos_cands = neighbors_sc[i]
                        if not pos_cands:
                            continue
                        neg_cands = [j for j in range(N_sc) if j != i and j not in set(pos_cands)]
                        if not neg_cands:
                            continue
                        anc_list.append(i); pos_list.append(random.choice(pos_cands)); neg_list.append(random.choice(neg_cands))
                    if anc_list:
                        a_t = torch.tensor(anc_list, dtype=torch.long, device=topo_vecs.device)
                        p_t = torch.tensor(pos_list, dtype=torch.long, device=topo_vecs.device)
                        n_t = torch.tensor(neg_list, dtype=torch.long, device=topo_vecs.device)
                        loss_sc = F.relu(sc_margin - (h_sc_norm[a_t]*h_sc_norm[p_t]).sum(1)
                                                    + (h_sc_norm[a_t]*h_sc_norm[n_t]).sum(1)).mean()
                        loss = loss + lambda_sc * loss_sc
            else:
                                     
                                                                         
                mem_size = self.TEM_vecs.shape[0] if not isinstance(self.TEM_vecs, list) else 0
                if mem_size >= 4:
                    k_sc_pos = int(args.tem_args.get('k_sc_pos', 3))
                    N_sc = min(N_sc_base, mem_size)
                    perm_sc = torch.randperm(mem_size, device=self.TEM_vecs.device)[:N_sc]
                    h_sc_in = self.TEM_vecs[perm_sc]
                    h_sc = h_sc_in
                    for layer_i, layer in enumerate(self.net.feat_trans_layers[:-1]):
                        h_sc = layer(h_sc)
                        if self.net.bn:
                            h_sc = self.net.bns[layer_i](h_sc)
                        h_sc = F.relu(h_sc)
                        h_sc = F.dropout(h_sc, p=self.net.dropout, training=self.training)
                    h_sc_norm = F.normalize(h_sc, p=2, dim=1)
                    with torch.no_grad():
                        sim_mat = torch.mm(F.normalize(h_sc_in, p=2, dim=1),
                                           F.normalize(h_sc_in, p=2, dim=1).t())
                    anc_list, pos_list, neg_list = [], [], []
                    for i in range(N_sc):
                        si = sim_mat[i].clone(); si[i] = -2.0
                        top_p = si.topk(min(k_sc_pos, N_sc - 1)).indices.tolist()
                        bot_n = si.topk(min(k_sc_pos, N_sc - 1), largest=False).indices.tolist()
                        anc_list.append(i); pos_list.append(random.choice(top_p)); neg_list.append(random.choice(bot_n))
                    a_t = torch.tensor(anc_list, dtype=torch.long, device=self.TEM_vecs.device)
                    p_t = torch.tensor(pos_list, dtype=torch.long, device=self.TEM_vecs.device)
                    n_t = torch.tensor(neg_list, dtype=torch.long, device=self.TEM_vecs.device)
                    loss_sc = F.relu(sc_margin - (h_sc_norm[a_t]*h_sc_norm[p_t]).sum(1)
                                                + (h_sc_norm[a_t]*h_sc_norm[n_t]).sum(1)).mean()
                    loss = loss + lambda_sc * loss_sc

        loss.backward()
        self.opt.step()

        if 'Hypergraph' in args.backbone:
                                                                 
            for key in list(g.ndata.keys()):
                g.ndata.pop(key)
        self._prev_task_cache = {
            'topo_vecs':         topo_vecs.detach(),
            'ids_per_cls_train': ids_per_cls_train,
            'labels_all':        labels_all,
            'graph':             g,
        }

        self.memory_audit.after_observe(args, t, self)

    def observe_task_IL(self, args, g, features, labels_all, t, train_ids, ids_per_cls, dataset):
        if not isinstance(self.TEM_vecs, list):
            self.TEM_vecs = []
            self.TEM_labels = []
        self.net.train()
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]
        topo_vecs = self.net.neighbor_agg(g, features)
        input_feat_concat = topo_vecs[train_ids]
        labels_concat = labels_all[train_ids]

        if t != self.current_task:
            self.current_task = t
            torch.cuda.empty_cache()                     
            if self._avg_deg is None:
                self._avg_deg, num_edges, num_nodes = _compute_avg_deg(g)
                # print(f"[SC Router][Task-IL] dataset={args.dataset}, avg_deg={self._avg_deg:.6f}, num_edges={num_edges}, num_nodes={num_nodes}")

            if self._prev_task_cache is not None:
                prev = self._prev_task_cache
                selected_ids = self.sampler(prev['ids_per_cls_train'], self.budget,
                                            neighbor_agg_model=self.net.neighbor_agg,
                                            graph=prev['graph'], topo_vecs=prev['topo_vecs'])
                self.TEM_vecs.append(prev['topo_vecs'][selected_ids].detach())
                self.TEM_labels.append(prev['labels_all'][selected_ids])

            self.memory_audit.snapshot('task_entry_after_memory_update', t, self)

        if args.cls_balance:
            n_per_cls = [(labels_concat == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        output, _ = self.net.feat_trans(g, input_feat_concat)
        loss = self.ce(output[:, offset1:offset2], labels_concat-offset1, weight=loss_w_[offset1: offset2])

        # memory replay
        if t > 0:
            for oldt in range(len(self.TEM_vecs)):
                o1, o2 = self.task_manager.get_label_offset(oldt-1)[1], self.task_manager.get_label_offset(oldt)[1]
                replay_sses = self.TEM_vecs[oldt]
                replay_labels = self.TEM_labels[oldt]
                if args.cls_balance:
                    n_per_cls = [(replay_labels == j).sum() for j in range(args.n_cls)]
                    loss_w_ = [1. / max(i, 1) for i in n_per_cls]
                else:
                    loss_w_ = [1. for i in range(args.n_cls)]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
                logits_replay, _ = self.net.feat_trans(g, replay_sses)
                loss_aux = self.ce(logits_replay[:,o1:o2], replay_labels-o1, weight=loss_w_[o1:o2])
                loss = loss + loss_aux

        lambda_sc = args.tem_args.get('lambda_sc', 0.0)
        if lambda_sc > 0.0:
            deg_tau = float(args.tem_args.get('deg_tau', 3.0))
            # print(f"[SC Router][Task-IL] avg_deg={self._avg_deg:.6f}, deg_tau={deg_tau:.6f}")
            use_graph_edge_sc = self._avg_deg >= deg_tau
            
            sc_margin = float(args.tem_args.get('sc_margin', 0.5))
            N_sc_base = int(args.tem_args.get('N_sc', 64))
            if use_graph_edge_sc:
                                        
                                                                       
                if len(train_ids) >= 4:
                    N_sc = min(N_sc_base, len(train_ids))
                    perm_sc = torch.randperm(len(train_ids))[:N_sc].tolist()
                    train_ids_list = train_ids if isinstance(train_ids, list) else train_ids.tolist()
                    sc_ids = [int(train_ids_list[i]) for i in perm_sc]
                    src_sc, dst_sc = g.edges()
                    src_sc, dst_sc = src_sc.tolist(), dst_sc.tolist()
                    sc_set = set(sc_ids)
                    sc_id_to_idx = {v: i for i, v in enumerate(sc_ids)}
                    neighbors_sc = {i: [] for i in range(N_sc)}
                    for s, d in zip(src_sc, dst_sc):
                        if s in sc_set and d in sc_set:
                            neighbors_sc[sc_id_to_idx[s]].append(sc_id_to_idx[d])
                            neighbors_sc[sc_id_to_idx[d]].append(sc_id_to_idx[s])
                    sc_tensor = torch.tensor(sc_ids, dtype=torch.long, device=topo_vecs.device)
                    h_sc = topo_vecs[sc_tensor]
                    k_top_nb = int(args.tem_args.get('k_top_nb', 0))
                    if k_top_nb > 0:
                        with torch.no_grad():
                            topo_nb_norm = F.normalize(h_sc, p=2, dim=1)
                            sim_nb = torch.mm(topo_nb_norm, topo_nb_norm.t())
                        for i in range(N_sc):
                            nbrs = neighbors_sc[i]
                            if len(nbrs) > k_top_nb:
                                neighbors_sc[i] = sorted(nbrs, key=lambda j: sim_nb[i, j].item(), reverse=True)[:k_top_nb]
                    for layer_i, layer in enumerate(self.net.feat_trans_layers[:-1]):
                        h_sc = layer(h_sc)
                        if self.net.bn:
                            h_sc = self.net.bns[layer_i](h_sc)
                        h_sc = F.relu(h_sc)
                        h_sc = F.dropout(h_sc, p=self.net.dropout, training=self.training)
                    h_sc_norm = F.normalize(h_sc, p=2, dim=1)
                    anc_list, pos_list, neg_list = [], [], []
                    for i in range(N_sc):
                        pos_cands = neighbors_sc[i]
                        if not pos_cands:
                            continue
                        neg_cands = [j for j in range(N_sc) if j != i and j not in set(pos_cands)]
                        if not neg_cands:
                            continue
                        anc_list.append(i); pos_list.append(random.choice(pos_cands)); neg_list.append(random.choice(neg_cands))
                    if anc_list:
                        a_t = torch.tensor(anc_list, dtype=torch.long, device=topo_vecs.device)
                        p_t = torch.tensor(pos_list, dtype=torch.long, device=topo_vecs.device)
                        n_t = torch.tensor(neg_list, dtype=torch.long, device=topo_vecs.device)
                        loss_sc = F.relu(sc_margin - (h_sc_norm[a_t]*h_sc_norm[p_t]).sum(1)
                                                    + (h_sc_norm[a_t]*h_sc_norm[n_t]).sum(1)).mean()
                        loss = loss + lambda_sc * loss_sc
            else:
                                     
                                                                         
                all_mem = torch.cat(self.TEM_vecs, dim=0) if (isinstance(self.TEM_vecs, list) and len(self.TEM_vecs) > 0) else None
                mem_size = all_mem.shape[0] if all_mem is not None else 0
                if mem_size >= 4:
                    k_sc_pos = int(args.tem_args.get('k_sc_pos', 3))
                    N_sc = min(N_sc_base, mem_size)
                    perm_sc = torch.randperm(mem_size, device=all_mem.device)[:N_sc]
                    h_sc_in = all_mem[perm_sc]
                    h_sc = h_sc_in
                    for layer_i, layer in enumerate(self.net.feat_trans_layers[:-1]):
                        h_sc = layer(h_sc)
                        if self.net.bn:
                            h_sc = self.net.bns[layer_i](h_sc)
                        h_sc = F.relu(h_sc)
                        h_sc = F.dropout(h_sc, p=self.net.dropout, training=self.training)
                    h_sc_norm = F.normalize(h_sc, p=2, dim=1)
                    with torch.no_grad():
                        sim_mat = torch.mm(F.normalize(h_sc_in, p=2, dim=1),
                                           F.normalize(h_sc_in, p=2, dim=1).t())
                    anc_list, pos_list, neg_list = [], [], []
                    for i in range(N_sc):
                        si = sim_mat[i].clone(); si[i] = -2.0
                        top_p = si.topk(min(k_sc_pos, N_sc - 1)).indices.tolist()
                        bot_n = si.topk(min(k_sc_pos, N_sc - 1), largest=False).indices.tolist()
                        anc_list.append(i); pos_list.append(random.choice(top_p)); neg_list.append(random.choice(bot_n))
                    a_t = torch.tensor(anc_list, dtype=torch.long, device=all_mem.device)
                    p_t = torch.tensor(pos_list, dtype=torch.long, device=all_mem.device)
                    n_t = torch.tensor(neg_list, dtype=torch.long, device=all_mem.device)
                    loss_sc = F.relu(sc_margin - (h_sc_norm[a_t]*h_sc_norm[p_t]).sum(1)
                                                + (h_sc_norm[a_t]*h_sc_norm[n_t]).sum(1)).mean()
                    loss = loss + lambda_sc * loss_sc

        loss.backward()
        self.opt.step()

        for key in list(g.ndata.keys()):
            g.ndata.pop(key)
        self._prev_task_cache = {
            'topo_vecs':         topo_vecs.detach(),
            'ids_per_cls_train': ids_per_cls_train,
            'labels_all':        labels_all,
            'graph':             g,
        }
        self.memory_audit.after_observe(args, t, self)
