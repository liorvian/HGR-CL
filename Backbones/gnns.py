# Contains DGL-derived aggregation code, modified through PDGNNs and HGR-CL.
# See LICENSES/DGL-Apache-2.0.txt and THIRD_PARTY_NOTICES.md.
from .gnnconv import GATConv, GCNLayer, GINConv
from .layers import PairNorm
import torch.nn.init as init
import random
from .utils import *
from dgl.base import DGLError
from dgl.nn.pytorch import edge_softmax
import dgl.function as fn
from dgl.nn.pytorch.conv import APPNPConv
import torch as th
import os
linear_choices = {'nn.Linear':nn.Linear, 'Linear_IL':Linear_IL}


def _trace_encoder(module, event, detail=''):
    """Opt-in call trace; no tensor operations or random-number consumption."""
    if os.environ.get('HSP_TRACE_ENCODER') != '1':
        return
    counts = getattr(module, '_encoder_trace_counts', None)
    if counts is None:
        counts = {}
        module._encoder_trace_counts = counts
    count = counts.get(event, 0) + 1
    counts[event] = count
    if count <= 3:
        print(f'[ENCODER_TRACE] {type(module).__name__}.{event} '
              f'call={count} {detail}', flush=True)
    elif count == 4:
        print(f'[ENCODER_TRACE] {type(module).__name__}.{event} '
              'further messages suppressed for this instance/event', flush=True)

class GIN(nn.Module):
    def __init__(self,
                 args,):
        super(GIN, self).__init__()
        dims = [args.d_data] + args.GIN_args['h_dims'] + [args.n_cls]
        self.dropout = args.GIN_args['dropout']
        self.gat_layers = nn.ModuleList()
        for l in range(len(dims)-1):
            lin = torch.nn.Linear(dims[l], dims[l+1])
            self.gat_layers.append(GINConv(lin, 'sum'))


    def forward(self, g, features):
        e_list = []
        h, e = self.gat_layers[0](g, features)
        x = F.relu(h)
        logits, e = self.gat_layers[1](g, x)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, features):
        e_list = []
        h, e = self.gat_layers[0].forward_batch(blocks[0], features)
        x = F.relu(h)
        logits, e = self.gat_layers[1].forward_batch(blocks[1], x)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

class GCN(nn.Module):
    def __init__(self,
                 args):
        super(GCN, self).__init__()
        dims = [args.d_data] + args.GCN_args['h_dims'] + [args.n_cls]
        self.dropout = args.GCN_args['dropout']
        self.gat_layers = nn.ModuleList()
        for l in range(len(dims)-1):
            self.gat_layers.append(GCNLayer(dims[l], dims[l+1]))

    def forward(self, g, features):
        e_list = []
        h = features
        for layer in self.gat_layers[:-1]:
            h, e = layer(g, h)
            h = F.relu(h)
            e_list = e_list + e
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits, e = self.gat_layers[-1](g, h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, features):
        e_list = []
        h = features
        for i,layer in enumerate(self.gat_layers[:-1]):
            h, e = layer.forward_batch(blocks[i], h)
            h = F.relu(h)
            e_list = e_list + e
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits, e = self.gat_layers[-1].forward_batch(blocks[-1], h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list


    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

class GAT(nn.Module):
    def __init__(self,
                 args,
                 heads,
                 activation):
        super(GAT, self).__init__()
        #self.g = g
        self.num_layers = args.GAT_args['num_layers']
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            args.d_data, args.GAT_args['num_hidden'], heads[0],
            args.GAT_args['feat_drop'], args.GAT_args['attn_drop'], args.GAT_args['negative_slope'], False, None))
        # self.norm_layers.append(nn.BatchNorm1d(num_hidden*heads[0]))
        self.norm_layers.append(PairNorm())
        
        # hidden layers
        for l in range(1, args.GAT_args['num_layers']):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                args.GAT_args['num_hidden'] * heads[l-1], args.GAT_args['num_hidden'], heads[l],
                args.GAT_args['feat_drop'], args.GAT_args['attn_drop'], args.GAT_args['negative_slope'], args.GAT_args['residual'], self.activation))
            # self.norm_layers.append(nn.BatchNorm1d(num_hidden*heads[l]))
            self.norm_layers.append(PairNorm())
        # output projection

        self.gat_layers.append(GATConv(
            args.GAT_args['num_hidden'] * heads[-2], args.n_cls, heads[-1],
            args.GAT_args['feat_drop'], args.GAT_args['attn_drop'], args.GAT_args['negative_slope'], args.GAT_args['residual'], None))

    def forward(self, g, inputs, save_logit_name = None):
        h = inputs
        e_list = []
        for l in range(self.num_layers):
            h, e = self.gat_layers[l](g, h)
            h = h.flatten(1)
            h = self.activation(h)
            e_list = e_list + e
        # store for ergnn
        self.second_last_h = h
        # output projection
        logits, e = self.gat_layers[-1](g, h)
        #self.second_last_h = logits if len(self.gat_layers) == 1 else h
        logits = logits.mean(1)
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, features):
        e_list = []
        h = features
        for i,layer in enumerate(self.gat_layers[:-1]):
            h, e = layer.forward_batch(blocks[i], h)
            h = h.flatten(1)
            h = self.activation(h)
            e_list = e_list + e
        logits, e = self.gat_layers[-1].forward_batch(blocks[-1], h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        logits = logits.mean(1)
        e_list = e_list + e
        return logits, e_list


    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()


class SGC_Agg(nn.Module):
    # only the neighborhood aggregation of SGC
    def __init__(self, k=1, cached=False, norm=None, allow_zero_in_degree=False):
        super().__init__()
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute Simplifying Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

        Note
        ----
        If ``cache`` is set to True, ``feat`` and ``graph`` should not change during
        training, or you will get wrong results.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)
                # compute (D^-1 A^k D)^k X
                for _ in range(self._k):
                    feat = feat * norm
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm

                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat
            # return self.fc(feat)
            return feat

    def forward_batch(self, blocks, feat):
        r"""

        Description
        -----------
        Compute Simplifying Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

        Note
        ----
        If ``cache`` is set to True, ``feat`` and ``graph`` should not change during
        training, or you will get wrong results.
        """
        if self._k != len(blocks):
            raise DGLError('The depth of the dataloader sampler is incompatible with the depth of SGC')
        for block in blocks:
            with block.local_scope():
                if not self._allow_zero_in_degree:
                    if (block.in_degrees() == 0).any():
                        raise DGLError('There are 0-in-degree nodes in the graph, '
                                       'output for those nodes will be invalid. '
                                       'This is harmful for some applications, '
                                       'causing silent performance regression. '
                                       'Adding self-loop on the input graph by '
                                       'calling `g = dgl.add_self_loop(g)` will resolve '
                                       'the issue. Setting ``allow_zero_in_degree`` '
                                       'to be `True` when constructing this module will '
                                       'suppress the check and let the code run.')

                if self._cached_h is not None:
                    feat = self._cached_h
                else:
                    # compute normalization
                    degs = block.out_degrees().float().clamp(min=1)
                    norm = th.pow(degs, -0.5)
                    norm = norm.to(feat.device).unsqueeze(1)
                    # compute (D^-1 A^k D)^k X
                    feat = feat * norm
                    block.srcdata['h'] = feat
                    block.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                    feat = block.dstdata.pop('h')
                    degs = block.in_degrees().float().clamp(min=1)
                    norm = th.pow(degs, -0.5)
                    norm = norm.to(feat.device).unsqueeze(1)
                    feat = feat * norm

        with blocks[-1].local_scope():
            if self.norm is not None:
                feat = self.norm(feat)

            # cache feature
            if self._cached:
                self._cached_h = feat

        # return self.fc(feat)
        return feat

class SGC(nn.Module):
    def __init__(self, args):
        super(SGC, self).__init__()
        linear_layer = linear_choices[args.SGC_args['linear']]
        if args.method == 'twp':
            self.twp=True
        else:
            self.twp=False
        self.bn = args.SGC_args['batch_norm']
        self.dropout = args.SGC_args['dropout']
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.gpu = args.gpu
        self.neighbor_agg = SGC_Agg(k=args.SGC_args['k'])
        self.feat_trans_layers = nn.ModuleList()
        if self.bn:
            self.bns = nn.ModuleList()
        h_dims = args.SGC_args['h_dims']
        if len(h_dims) > 0:
            self.feat_trans_layers.append(linear_layer(args.d_data, h_dims[0], bias=args.SGC_args['linear_bias']))
            if self.bn:
                self.bns.append(nn.BatchNorm1d(h_dims[0]))
            for i in range(len(h_dims) - 1):
                self.feat_trans_layers.append(linear_layer(h_dims[i], h_dims[i + 1], bias=args.SGC_args['linear_bias']))
                if self.bn:
                    self.bns.append(nn.BatchNorm1d(h_dims[i + 1]))
            self.feat_trans_layers.append(linear_layer(h_dims[-1], args.n_cls, bias=args.SGC_args['linear_bias']))
        elif len(h_dims) == 0:
            self.feat_trans_layers.append(linear_layer(args.d_data, args.n_cls, bias=args.SGC_args['linear_bias']))
        else:
            raise ValueError('no valid MLP dims are given')

    def forward(self, graph, x, twp=False, tasks=None):
        graph = graph.local_var().to('cuda:{}'.format(self.gpu))
        e_list = []
        x = self.neighbor_agg(graph, x)
        logits, e = self.feat_trans(graph, x, twp=twp, cls=tasks)
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, x, twp=False, tasks=None):
        #graph = graph.local_var().to('cuda:{}'.format(self.gpu))
        e_list = []
        x = self.neighbor_agg.forward_batch(blocks, x)
        logits, e = self.feat_trans(blocks[0], x, twp=twp, cls=tasks)
        e_list = e_list + e
        return logits, e_list

    def feat_trans(self, graph, x, twp=False, cls=None):
        for i, layer in enumerate(self.feat_trans_layers[:-1]):
            x = layer(x)
            if self.bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.feat_trans_layers[-1](x)

        self.second_last_h = x

        mask = torch.zeros(x.shape[-1], device=x.get_device())
        if cls is not None:
            mask[cls] = 1.
        else:
            mask[:] = 1.
        x = x * mask
        # for twp
        elist = []
        if self.twp:
            graph.srcdata['h'] = x
            graph.apply_edges(
                lambda edges: {'e': torch.sum((torch.mul(edges.src['h'], torch.tanh(edges.dst['h']))), 1)})
            e = self.leaky_relu(graph.edata.pop('e'))
            e_soft = edge_softmax(graph, e)

            elist.append(e_soft)

        return x, elist
        #return x.log_softmax(dim=-1), elist
    def reset_params(self):
        for layer in self.feat_trans_layers:
            layer.reset_parameters()

class CustomDecoupledAPPNP(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.method == 'twp':
            self.twp=True
        else:
            self.twp=False
        self.gpu = args.gpu
        ags = args.CustomDecoupledAPPNP_args
        h_dims, dropout, bias, k, alpha, self.bn = ags['h_dims'], ags['dropout'], ags['bias'], ags['k'], ags['alpha'], ags['batch_norm']
        mlp_dims = [args.d_data] + h_dims + [args.n_cls]

        linear_layer = linear_choices[ags['linear']]

        fcs = []
        for i in range(1, len(mlp_dims)):
            fcs.append(linear_layer(mlp_dims[i - 1], mlp_dims[i], bias=bias))
        self.fcs = nn.ModuleList(fcs)
        self.reg_params = list(self.fcs[0].parameters())
        self.dropout = nn.Dropout(p=dropout)
        self.act_fn = nn.ReLU()
        self.neighbor_agg = APPNPConv(k, alpha)

    def feat_trans(self, graph, x, twp=False, cls=None):
        for fc in self.fcs[0:-1]:
            x = self.act_fn(fc(self.dropout(x)))
        x = self.fcs[-1](self.dropout(x))
        self.second_last_h = x
        # for twp
        elist = []
        if self.twp:
            graph.srcdata['h'] = x
            graph.apply_edges(
                lambda edges: {'e': torch.sum((torch.mul(edges.src['h'], torch.tanh(edges.dst['h']))), 1)})
            e = self.leaky_relu(graph.edata.pop('e'))
            e_soft = edge_softmax(graph, e)

            elist.append(e_soft)

        return x, elist

    def forward(self, graph, x, twp=False, tasks=None):
        graph = graph.local_var().to('cuda:{}'.format(self.gpu))
        e_list = []
        x = self.neighbor_agg(graph, x)
        logits, e = self.feat_trans(graph, x, twp=twp, cls=tasks)
        e_list = e_list + e
        return logits, e_list

    def reset_params(self):
        for layer in self.fcs:
            layer.reset_parameters()

class CustomS2GC_Agg(nn.Module):
    # only the neighborhood aggregation of S2GC
    def __init__(self,
                 alpha=0.05,
                 k=1,
                 cached=False,
                 norm=None,
                 allow_zero_in_degree=False):
        super().__init__()
        self._cached = cached
        self._cached_h = None
        self._alpha = alpha
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)
                # compute SUM_k(D^-1 A^k D)^k X
                feat_ori = copy.deepcopy(feat)
                feat_sum_k = 0
                for _ in range(self._k):
                    feat = feat * norm
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'),
                                     fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm
                    feat_sum_k+=feat
                feat = ((1.-self._alpha)/self._k)*feat_sum_k + self._alpha*feat_ori

                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat
            #return self.fc(feat)
            return feat

class CustomDecoupledS2GC(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.method == 'twp':
            self.twp=True
        else:
            self.twp=False
        self.gpu = args.gpu
        ags = args.CustomDecoupledS2GC_args
        linear_layer = linear_choices[ags['linear']]
        h_dims, self.dropout, bias, k, alpha, self.bn = ags['h_dims'], ags['dropout'], ags['bias'], ags['k'], ags['alpha'], ags['batch_norm']
        self.neighbor_agg = CustomS2GC_Agg(alpha=alpha, k=k)
        self.feat_trans_layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        dims = [args.d_data] + h_dims + [args.n_cls]

        fcs = []
        for i in range(1, len(dims)-1):
            fcs.append(linear_layer(dims[i - 1], dims[i], bias=bias))
            self.bns.append(nn.BatchNorm1d(dims[i]))
        fcs.append(linear_layer(dims[- 2], dims[-1], bias=bias))
        self.feat_trans_layers = nn.ModuleList(fcs)

    def feat_trans(self, graph, x, twp=False, cls=None):
        for i, layer in enumerate(self.feat_trans_layers[:-1]):
            x = layer(x)
            if self.bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.feat_trans_layers[-1](x)

        self.second_last_h = x

        mask = torch.zeros(x.shape[-1], device=x.get_device())
        if cls is not None:
            mask[cls] = 1.
        else:
            mask[:] = 1.
        x = x * mask

        # for twp
        elist = []
        if self.twp:
            graph.srcdata['h'] = x
            graph.apply_edges(
                lambda edges: {'e': torch.sum((torch.mul(edges.src['h'], torch.tanh(edges.dst['h']))), 1)})
            e = self.leaky_relu(graph.edata.pop('e'))
            e_soft = edge_softmax(graph, e)

            elist.append(e_soft)

        return x, elist

    def forward(self, graph, x, twp=False, tasks=None):
        graph = graph.local_var().to('cuda:{}'.format(self.gpu))
        e_list = []
        x = self.neighbor_agg(graph, x)
        logits, e = self.feat_trans(graph, x, twp=twp, cls=tasks)
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, x, tasks=None):
        x = self.neighbor_agg(blocks[-1], x)
        x = self.feat_trans(x, tasks)
        return x

    def reset_params(self):
        for layer in self.feat_trans_layers:
            layer.reset_parameters()

class CustomSGC_Agg(nn.Module):
    # only the neighborhood aggregation of SGC
    def __init__(self, k=1, cached=False, norm=None, allow_zero_in_degree=False):
        super().__init__()
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute Simplifying Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

        Note
        ----
        If ``cache`` is set to True, ``feat`` and ``graph`` should not change during
        training, or you will get wrong results.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)
                # compute (D^-1 A^k D)^k X
                for _ in range(self._k):
                    feat = feat * norm
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm

                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat
            # return self.fc(feat)
            return feat

class CustomDecoupledSGC(nn.Module):
    def __init__(self, args):
        super(CustomDecoupledSGC, self).__init__()
        linear_layer = linear_choices[args.CustomDecoupledSGC_args['linear']]
        if args.method == 'twp':
            self.twp=True
        else:
            self.twp=False
        self.bn = args.CustomDecoupledSGC_args['batch_norm']
        self.dropout = args.CustomDecoupledSGC_args['dropout']
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.gpu = args.gpu
        self.neighbor_agg = CustomSGC_Agg(k=args.CustomDecoupledSGC_args['k'])
        self.feat_trans_layers = nn.ModuleList()
        if self.bn:
            self.bns = nn.ModuleList()
        h_dims = args.CustomDecoupledSGC_args['h_dims']
        if len(h_dims) > 0:
            self.feat_trans_layers.append(linear_layer(args.d_data, h_dims[0], bias=args.CustomDecoupledSGC_args['linear_bias']))
            if self.bn:
                self.bns.append(nn.BatchNorm1d(h_dims[0]))
            for i in range(len(h_dims) - 1):
                self.feat_trans_layers.append(linear_layer(h_dims[i], h_dims[i + 1], bias=args.CustomDecoupledSGC_args['linear_bias']))
                if self.bn:
                    self.bns.append(nn.BatchNorm1d(h_dims[i + 1]))
            self.feat_trans_layers.append(linear_layer(h_dims[-1], args.n_cls, bias=args.CustomDecoupledSGC_args['linear_bias']))
        elif len(h_dims) == 0:
            self.feat_trans_layers.append(linear_layer(args.d_data, args.n_cls, bias=args.CustomDecoupledSGC_args['linear_bias']))
        else:
            raise ValueError('no valid MLP dims are given')

    def forward(self, graph, x, twp=False, tasks=None):
        graph = graph.local_var().to('cuda:{}'.format(self.gpu))
        e_list = []
        x = self.neighbor_agg(graph, x)
        logits, e = self.feat_trans(graph, x, twp=twp, cls=tasks)
        e_list = e_list + e
        return logits, e_list

    def feat_trans(self, graph, x, twp=False, cls=None):
        for i, layer in enumerate(self.feat_trans_layers[:-1]):
            x = layer(x)
            if self.bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.feat_trans_layers[-1](x)

        self.second_last_h = x

        mask = torch.zeros(x.shape[-1], device=x.get_device())
        if cls is not None:
            mask[cls] = 1.
        else:
            mask[:] = 1.
        x = x * mask
        # for twp
        elist = []
        if self.twp:
            graph.srcdata['h'] = x
            graph.apply_edges(
                lambda edges: {'e': torch.sum((torch.mul(edges.src['h'], torch.tanh(edges.dst['h']))), 1)})
            e = self.leaky_relu(graph.edata.pop('e'))
            e_soft = edge_softmax(graph, e)

            elist.append(e_soft)

        return x, elist
        #return x.log_softmax(dim=-1), elist
    def reset_params(self):
        for layer in self.feat_trans_layers:
            layer.reset_parameters()

class CustomFDGNN_Agg(nn.Module):
    # only the neighborhood aggregation of CustomFDGNN
    def __init__(self, dim, k=1, cached=False, norm=None, allow_zero_in_degree=False):
        super().__init__()
        #self.dims = dims
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.W_center = []
        self.W_nb = []
        for l in range(1):
            self.W_center.append(init.kaiming_uniform_(torch.Tensor(dim, dim)))
            self.W_nb.append(init.kaiming_uniform_(torch.Tensor(dim, dim)))

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute Simplifying Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

        Note
        ----
        If ``cache`` is set to True, ``feat`` and ``graph`` should not change during
        training, or you will get wrong results.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)
                # compute (D^-1 A^k D)^k X
                for _ in range(self._k):
                    feat = feat * norm
                    graph.ndata['h'] = feat.mm(self.W_nb[_].cuda(feat.get_device()))
                    graph.ndata['center'] = feat.mm(self.W_center[_].cuda(feat.get_device()))
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'center'))
                    feat = graph.ndata.pop('center')
                    feat = feat * norm

                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat
            # return self.fc(feat)
            return feat


class HypergraphSGC_Agg(nn.Module):
    def __init__(self, k=2, max_hyperedge_size=32):
        super().__init__()
        self._k = k
        self._max_hyperedge_size = max_hyperedge_size
        self._stats_printed = False
        
        self._cache_graph_id = None
        self._cache_g = None
        self._cache_H = None
        self._cache_H_t = None
        self._cache_Dv_invsqrt = None
        self._cache_De_inv = None

    def _build_incidence(self, graph, device):
        """Build the graph incidence matrix."""
        _trace_encoder(self, '_build_incidence',
                       f'symmetrize=True max_hyperedge_size={self._max_hyperedge_size}')
        n_nodes = graph.num_nodes()
        src, dst = graph.edges()
        src, dst = src.tolist(), dst.tolist()

        
        neighbors = {v: set() for v in range(n_nodes)}
        for s, d in zip(src, dst):
            neighbors[d].add(s)
            neighbors[s].add(d)

        row_ids = []
        col_ids = []
        hyperedge_sizes = []
        max_he = self._max_hyperedge_size

        for v in range(n_nodes):
            nodes_in_hedge = list(neighbors[v] | {v})
            
            if len(nodes_in_hedge) > max_he:
                others = [u for u in nodes_in_hedge if u != v]
                sampled = random.sample(others, max_he - 1)
                nodes_in_hedge = sampled + [v]
            hyperedge_sizes.append(len(nodes_in_hedge))
            for u in nodes_in_hedge:
                row_ids.append(u)
                col_ids.append(v)

        row_tensor = torch.tensor(row_ids, dtype=torch.long, device=device)
        col_tensor = torch.tensor(col_ids, dtype=torch.long, device=device)
        values = torch.ones(len(row_ids), dtype=torch.float32, device=device)
        
        H = torch.sparse_coo_tensor(
            torch.stack([row_tensor, col_tensor]),
            values, (n_nodes, n_nodes), device=device
        ).coalesce()
        H_t = torch.sparse_coo_tensor(
            torch.stack([col_tensor, row_tensor]),
            values, (n_nodes, n_nodes), device=device
        ).coalesce()
        return H, H_t, hyperedge_sizes

    def _print_stats(self, n_nodes, hyperedge_sizes):
        n_edges = len(hyperedge_sizes)
        avg_size = sum(hyperedge_sizes) / n_edges if n_edges > 0 else 0.0
        max_size = max(hyperedge_sizes) if n_edges > 0 else 0

    def forward(self, graph, feat):
        device = feat.device
        graph_id = (graph.num_nodes(), graph.num_edges())

        _trace_encoder(self, 'forward',
                       f'nodes={graph_id[0]} edges={graph_id[1]} K={self._k} '
                       f'cache_hit={self._cache_graph_id == graph_id}')

        
            
        if self._cache_graph_id != graph_id:
            if self._cache_g is not None:
                for key in list(self._cache_g.ndata.keys()):
                    self._cache_g.ndata.pop(key)
                self._cache_g = None
            
            g = dgl.add_self_loop(graph).to(device)
            _trace_encoder(self, 'cache_rebuild', 'operation=add_self_loop')
            
            deg = g.in_degrees().float().to(device).clamp(min=1)
            self._cache_Dv_invsqrt = deg.pow(-0.5).unsqueeze(1)
            self._cache_De_inv     = deg.pow(-1).unsqueeze(1)
            self._cache_g          = g
            self._cache_graph_id   = graph_id
        g           = self._cache_g
        Dv_invsqrt  = self._cache_Dv_invsqrt
        De_inv      = self._cache_De_inv

        
        
        for _ in range(self._k):
            feat = feat * Dv_invsqrt                        # D_v^{-1/2} X
            g.ndata['h'] = feat
            _trace_encoder(self, 'aggregation_1', 'operation=update_all graph=cached_g')
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'agg'))
            feat = g.ndata['agg'] * De_inv                  # D_e^{-1} (A+I) D_v^{-1/2} X
            g.ndata['h'] = feat
            _trace_encoder(self, 'aggregation_2', 'operation=update_all graph=cached_g')
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'agg'))
            feat = g.ndata['agg'] * Dv_invsqrt              # D_v^{-1/2} (A+I) D_e^{-1} (A+I) D_v^{-1/2} X

        
        g.ndata.pop('agg', None)
        return feat


class CustomDecoupledHypergraphSGC(nn.Module):
    def __init__(self, args):
        super().__init__()
        linear_layer = linear_choices[args.CustomDecoupledHypergraphSGC_args['linear']]
        if args.method == 'twp':
            self.twp = True
        else:
            self.twp = False
        self.bn = args.CustomDecoupledHypergraphSGC_args['batch_norm']
        self.dropout = args.CustomDecoupledHypergraphSGC_args['dropout']
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.gpu = args.gpu
        self.neighbor_agg = HypergraphSGC_Agg(
            k=args.CustomDecoupledHypergraphSGC_args['k'],
            max_hyperedge_size=args.CustomDecoupledHypergraphSGC_args['max_hyperedge_size'],
        )
        self.feat_trans_layers = nn.ModuleList()
        if self.bn:
            self.bns = nn.ModuleList()
        h_dims = args.CustomDecoupledHypergraphSGC_args['h_dims']
        if len(h_dims) > 0:
            self.feat_trans_layers.append(linear_layer(args.d_data, h_dims[0], bias=args.CustomDecoupledHypergraphSGC_args['linear_bias']))
            if self.bn:
                self.bns.append(nn.BatchNorm1d(h_dims[0]))
            for i in range(len(h_dims) - 1):
                self.feat_trans_layers.append(linear_layer(h_dims[i], h_dims[i + 1], bias=args.CustomDecoupledHypergraphSGC_args['linear_bias']))
                if self.bn:
                    self.bns.append(nn.BatchNorm1d(h_dims[i + 1]))
            self.feat_trans_layers.append(linear_layer(h_dims[-1], args.n_cls, bias=args.CustomDecoupledHypergraphSGC_args['linear_bias']))
        elif len(h_dims) == 0:
            self.feat_trans_layers.append(linear_layer(args.d_data, args.n_cls, bias=args.CustomDecoupledHypergraphSGC_args['linear_bias']))
        else:
            raise ValueError('no valid MLP dims are given')

    def forward(self, graph, x, twp=False, tasks=None):
        _trace_encoder(self, 'forward', f'delegate={type(self.neighbor_agg).__name__}')
        graph = graph.local_var().to('cuda:{}'.format(self.gpu))
        e_list = []
        x = self.neighbor_agg(graph, x)
        logits, e = self.feat_trans(graph, x, twp=twp, cls=tasks)
        e_list = e_list + e
        return logits, e_list

    def feat_trans(self, graph, x, twp=False, cls=None):
        for i, layer in enumerate(self.feat_trans_layers[:-1]):
            x = layer(x)
            if self.bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.feat_trans_layers[-1](x)

        self.second_last_h = x

        mask = torch.zeros(x.shape[-1], device=x.get_device())
        if cls is not None:
            mask[cls] = 1.
        else:
            mask[:] = 1.
        x = x * mask

        elist = []
        if self.twp:
            graph.srcdata['h'] = x
            graph.apply_edges(
                lambda edges: {'e': torch.sum((torch.mul(edges.src['h'], torch.tanh(edges.dst['h']))), 1)})
            e = self.leaky_relu(graph.edata.pop('e'))
            e_soft = edge_softmax(graph, e)
            elist.append(e_soft)

        return x, elist

    def reset_params(self):
        for layer in self.feat_trans_layers:
            layer.reset_parameters()


class HypergraphPlusSGC_Agg(nn.Module):
    def __init__(self, k=2, max_hyperedge_size=32):
        super().__init__()
        self._k = k
        self._max_hyperedge_size = max_hyperedge_size
        self._stats_printed = False
        self._cache_graph_id   = None
        self._cache_H          = None   # [N, N] sparse COO
        self._cache_H_t        = None   # [N, N] sparse COO (transpose)
        self._cache_Dv_invsqrt = None   # [N] dense
        self._cache_De_inv     = None   # [N] dense

    def _build_incidence(self, graph, device):
        """Build the hyperedge incidence matrix."""
        _trace_encoder(self, '_build_incidence',
                       f'symmetrize=True max_hyperedge_size={self._max_hyperedge_size}')
        import random as _random
        n_nodes = graph.num_nodes()
        src, dst = graph.edges()
        src, dst = src.tolist(), dst.tolist()

        neighbors = {v: set() for v in range(n_nodes)}
        for s, d in zip(src, dst):
            neighbors[d].add(s)
            neighbors[s].add(d)

        row_ids, col_ids = [], []
        max_he = self._max_hyperedge_size

        for v in range(n_nodes):
            nodes_in_hedge = list(neighbors[v] | {v})
            if len(nodes_in_hedge) > max_he:
                others = [u for u in nodes_in_hedge if u != v]
                sampled = _random.sample(others, max_he - 1)
                nodes_in_hedge = sampled + [v]
            for u in nodes_in_hedge:
                row_ids.append(u)
                col_ids.append(v)

        row_t = torch.tensor(row_ids, dtype=torch.long, device=device)
        col_t = torch.tensor(col_ids, dtype=torch.long, device=device)
        vals  = torch.ones(len(row_ids), dtype=torch.float32, device=device)

        H   = torch.sparse_coo_tensor(torch.stack([row_t, col_t]),
                                      vals, (n_nodes, n_nodes)).coalesce()
        H_t = torch.sparse_coo_tensor(torch.stack([col_t, row_t]),
                                      vals, (n_nodes, n_nodes)).coalesce()
        return H, H_t

    def forward(self, graph, feat):
        device    = feat.device
        graph_id  = (graph.num_nodes(), graph.num_edges())

        _trace_encoder(self, 'forward',
                       f'nodes={graph_id[0]} edges={graph_id[1]} K={self._k} '
                       f'cache_hit={self._cache_graph_id == graph_id}')

        if self._cache_graph_id != graph_id:
            H, H_t = self._build_incidence(graph, device)
            _trace_encoder(self, 'cache_rebuild', 'operation=build_H_and_H_transpose')

            
            
            De = torch.sparse.sum(H, dim=0).to_dense().clamp(min=1)

            self._cache_H          = H
            self._cache_H_t        = H_t
            self._cache_Dv_invsqrt = Dv.pow(-0.5).unsqueeze(1)   # [N, 1]
            self._cache_De_inv     = De.pow(-1).unsqueeze(1)      # [N, 1]
            self._cache_graph_id   = graph_id

        H          = self._cache_H
        H_t        = self._cache_H_t
        Dv_invsqrt = self._cache_Dv_invsqrt
        De_inv     = self._cache_De_inv

        
        for _ in range(self._k):
            feat = feat * Dv_invsqrt                        # D_v^{-1/2} X
            _trace_encoder(self, 'aggregation_1', 'operation=sparse_mm matrix=H_transpose')
            feat = torch.sparse.mm(H_t, feat)               # H^T D_v^{-1/2} X
            feat = feat * De_inv                            # D_e^{-1} H^T D_v^{-1/2} X
            _trace_encoder(self, 'aggregation_2', 'operation=sparse_mm matrix=H')
            feat = torch.sparse.mm(H, feat)                # H D_e^{-1} H^T D_v^{-1/2} X
            feat = feat * Dv_invsqrt                        # D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2} X

        return feat


class CustomDecoupledHypergraphPlusSGC(nn.Module):
    def __init__(self, args):
        super().__init__()
        linear_layer = linear_choices[args.CustomDecoupledHypergraphSGC_args['linear']]
        if args.method == 'twp':
            self.twp = True
        else:
            self.twp = False
        self.bn       = args.CustomDecoupledHypergraphSGC_args['batch_norm']
        self.dropout  = args.CustomDecoupledHypergraphSGC_args['dropout']
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.gpu      = args.gpu
        self.neighbor_agg = HypergraphPlusSGC_Agg(
            k=args.CustomDecoupledHypergraphSGC_args['k'],
            max_hyperedge_size=args.CustomDecoupledHypergraphSGC_args['max_hyperedge_size'],
        )
        self.feat_trans_layers = nn.ModuleList()
        if self.bn:
            self.bns = nn.ModuleList()
        h_dims = args.CustomDecoupledHypergraphSGC_args['h_dims']
        if len(h_dims) > 0:
            self.feat_trans_layers.append(
                linear_layer(args.d_data, h_dims[0],
                             bias=args.CustomDecoupledHypergraphSGC_args['linear_bias']))
            if self.bn:
                self.bns.append(nn.BatchNorm1d(h_dims[0]))
            for i in range(len(h_dims) - 1):
                self.feat_trans_layers.append(
                    linear_layer(h_dims[i], h_dims[i + 1],
                                 bias=args.CustomDecoupledHypergraphSGC_args['linear_bias']))
                if self.bn:
                    self.bns.append(nn.BatchNorm1d(h_dims[i + 1]))
            self.feat_trans_layers.append(
                linear_layer(h_dims[-1], args.n_cls,
                             bias=args.CustomDecoupledHypergraphSGC_args['linear_bias']))
        elif len(h_dims) == 0:
            self.feat_trans_layers.append(
                linear_layer(args.d_data, args.n_cls,
                             bias=args.CustomDecoupledHypergraphSGC_args['linear_bias']))
        else:
            raise ValueError('no valid MLP dims are given')

    def forward(self, graph, x, twp=False, tasks=None):
        _trace_encoder(self, 'forward', f'delegate={type(self.neighbor_agg).__name__}')
        graph  = graph.local_var().to('cuda:{}'.format(self.gpu))
        e_list = []
        x      = self.neighbor_agg(graph, x)
        logits, e = self.feat_trans(graph, x, twp=twp, cls=tasks)
        e_list    = e_list + e
        return logits, e_list

    def feat_trans(self, graph, x, twp=False, cls=None):
        for i, layer in enumerate(self.feat_trans_layers[:-1]):
            x = layer(x)
            if self.bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.feat_trans_layers[-1](x)

        self.second_last_h = x

        mask = torch.zeros(x.shape[-1], device=x.get_device())
        if cls is not None:
            mask[cls] = 1.
        else:
            mask[:] = 1.
        x = x * mask

        elist = []
        if self.twp:
            graph.srcdata['h'] = x
            graph.apply_edges(
                lambda edges: {'e': torch.sum(
                    (torch.mul(edges.src['h'], torch.tanh(edges.dst['h']))), 1)})
            e     = self.leaky_relu(graph.edata.pop('e'))
            e_soft = edge_softmax(graph, e)
            elist.append(e_soft)

        return x, elist

    def reset_params(self):
        for layer in self.feat_trans_layers:
            layer.reset_parameters()


class CliqueExpandSGC_Agg(nn.Module):
    def __init__(self, k=2, max_hyperedge_size=32):
        super().__init__()
        self._k = k
        self._max_he = max_hyperedge_size
        self._cache_graph_id = None
        self._cache_g = None
        self._cache_norm = None
        self._stats_printed = False

    def forward(self, graph, feat):
        device = feat.device
        graph_id = (graph.num_nodes(), graph.num_edges())

        if self._cache_graph_id != graph_id:
            if self._cache_g is not None:
                for key in list(self._cache_g.ndata.keys()):
                    self._cache_g.ndata.pop(key)
                self._cache_g = None
            from .hypergraph_utils import build_expanded_graph
            g_exp, hyperedges = build_expanded_graph(
                graph, mode='augment',
                max_hyperedge_size=self._max_he
            )
            g = g_exp.to(device)
            deg = g.in_degrees().float().to(device).clamp(min=1)
            self._cache_norm = deg.pow(-0.5).unsqueeze(1)
            self._cache_g = g
            self._cache_graph_id = graph_id
        g, norm = self._cache_g, self._cache_norm
        for _ in range(self._k):
            feat = feat * norm
            g.ndata['h'] = feat
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'agg'))
            feat = g.ndata.pop('agg') * norm
        g.ndata.pop('h', None)
        return feat


class CustomDecoupledCliqueExpandSGC(nn.Module):
    def __init__(self, args):
        super().__init__()
        linear_layer = linear_choices[args.CustomDecoupledHypergraphSGC_args['linear']]
        self.twp = args.method == 'twp'
        self.bn = args.CustomDecoupledHypergraphSGC_args['batch_norm']
        self.dropout = args.CustomDecoupledHypergraphSGC_args['dropout']
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.gpu = args.gpu
        self.neighbor_agg = CliqueExpandSGC_Agg(
            k=args.CustomDecoupledHypergraphSGC_args['k'],
            max_hyperedge_size=args.CustomDecoupledHypergraphSGC_args['max_hyperedge_size'],
        )
        self.feat_trans_layers = nn.ModuleList()
        if self.bn:
            self.bns = nn.ModuleList()
        h_dims = args.CustomDecoupledHypergraphSGC_args['h_dims']
        if len(h_dims) > 0:
            self.feat_trans_layers.append(
                linear_layer(args.d_data, h_dims[0],
                             bias=args.CustomDecoupledHypergraphSGC_args['linear_bias']))
            if self.bn:
                self.bns.append(nn.BatchNorm1d(h_dims[0]))
            for i in range(len(h_dims) - 1):
                self.feat_trans_layers.append(
                    linear_layer(h_dims[i], h_dims[i + 1],
                                 bias=args.CustomDecoupledHypergraphSGC_args['linear_bias']))
                if self.bn:
                    self.bns.append(nn.BatchNorm1d(h_dims[i + 1]))
            self.feat_trans_layers.append(
                linear_layer(h_dims[-1], args.n_cls,
                             bias=args.CustomDecoupledHypergraphSGC_args['linear_bias']))
        else:
            self.feat_trans_layers.append(
                linear_layer(args.d_data, args.n_cls,
                             bias=args.CustomDecoupledHypergraphSGC_args['linear_bias']))

    def forward(self, graph, x, twp=False, tasks=None):
        graph = graph.local_var().to('cuda:{}'.format(self.gpu))
        x = self.neighbor_agg(graph, x)
        logits, e = self.feat_trans(graph, x, twp=twp, cls=tasks)
        return logits, e

    def feat_trans(self, graph, x, twp=False, cls=None):
        for i, layer in enumerate(self.feat_trans_layers[:-1]):
            x = layer(x)
            if self.bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.feat_trans_layers[-1](x)
        self.second_last_h = x
        mask = torch.zeros(x.shape[-1], device=x.get_device())
        if cls is not None:
            mask[cls] = 1.
        else:
            mask[:] = 1.
        x = x * mask
        elist = []
        if self.twp:
            graph.srcdata['h'] = x
            graph.apply_edges(
                lambda edges: {'e': torch.sum(
                    (torch.mul(edges.src['h'], torch.tanh(edges.dst['h']))), 1)})
            e = self.leaky_relu(graph.edata.pop('e'))
            elist.append(edge_softmax(graph, e))
        return x, elist

    def reset_params(self):
        for layer in self.feat_trans_layers:
            layer.reset_parameters()


class CustomFDGNN(nn.Module):
    # a reservoir computing module mentioned by the reviewer
    def __init__(self, args):
        super().__init__()
        linear_layer = linear_choices[args.CustomFDGNN_args['linear']]
        if args.method == 'twp':
            self.twp=True
        else:
            self.twp=False
        self.bn = args.CustomFDGNN_args['batch_norm']
        self.dropout = args.CustomFDGNN_args['dropout']
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.gpu = args.gpu
        self.neighbor_agg = CustomFDGNN_Agg(args.d_data,k=args.CustomFDGNN_args['k'])
        self.feat_trans_layers = nn.ModuleList()
        if self.bn:
            self.bns = nn.ModuleList()
        h_dims = args.CustomFDGNN_args['h_dims']
        if len(h_dims) > 0:
            self.feat_trans_layers.append(linear_layer(args.d_data, h_dims[0], bias=args.CustomFDGNN_args['linear_bias']))
            if self.bn:
                self.bns.append(nn.BatchNorm1d(h_dims[0]))
            for i in range(len(h_dims) - 1):
                self.feat_trans_layers.append(linear_layer(h_dims[i], h_dims[i + 1], bias=args.CustomFDGNN_args['linear_bias']))
                if self.bn:
                    self.bns.append(nn.BatchNorm1d(h_dims[i + 1]))
            self.feat_trans_layers.append(linear_layer(h_dims[-1], args.n_cls, bias=args.CustomFDGNN_args['linear_bias']))
        elif len(h_dims) == 0:
            self.feat_trans_layers.append(linear_layer(args.d_data, args.n_cls, bias=args.CustomFDGNN_args['linear_bias']))
        else:
            raise ValueError('no valid MLP dims are given')

    def forward(self, graph, x, twp=False, tasks=None):
        graph = graph.local_var().to('cuda:{}'.format(self.gpu))
        e_list = []
        x = self.neighbor_agg(graph, x)
        logits, e = self.feat_trans(graph, x, twp=twp, cls=tasks)
        e_list = e_list + e
        return logits, e_list

    def feat_trans(self, graph, x, twp=False, cls=None):
        for i, layer in enumerate(self.feat_trans_layers[:-1]):
            x = layer(x)
            if self.bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.feat_trans_layers[-1](x)

        self.second_last_h = x

        mask = torch.zeros(x.shape[-1], device=x.get_device())
        if cls is not None:
            mask[cls] = 1.
        else:
            mask[:] = 1.
        x = x * mask
        # for twp
        elist = []
        if self.twp:
            graph.srcdata['h'] = x
            graph.apply_edges(
                lambda edges: {'e': torch.sum((torch.mul(edges.src['h'], torch.tanh(edges.dst['h']))), 1)})
            e = self.leaky_relu(graph.edata.pop('e'))
            e_soft = edge_softmax(graph, e)

            elist.append(e_soft)

        return x, elist
        #return x.log_softmax(dim=-1), elist
    def reset_params(self):
        for layer in self.feat_trans_layers:
            layer.reset_parameters()

