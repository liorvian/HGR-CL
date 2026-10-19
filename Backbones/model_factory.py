import torch.nn.functional as F
from .gnns import GAT, GCN, GIN, CustomDecoupledSGC, CustomDecoupledS2GC, CustomDecoupledAPPNP, CustomFDGNN, CustomDecoupledHypergraphSGC, CustomDecoupledHypergraphPlusSGC, CustomDecoupledCliqueExpandSGC

def get_model(dataset, args):
    n_classes = args.n_cls_per_task
    print('n_classes', n_classes)
    if args.backbone == 'GAT':
        heads = ([args.GAT_args['heads']] * args.GAT_args['num_layers']) + [args.GAT_args['out_heads']]
        model = GAT(args, heads, F.elu)
    elif args.backbone == 'GCN':
        model = GCN(args)
    elif args.backbone == 'GIN':
        model = GIN(args)
    elif args.backbone in ['CustomDecoupledSGC', 'CustomDecoupledS2GC', 'CustomDecoupledAPPNP', 'CustomFDGNN', 'CustomDecoupledHypergraphSGC', 'CustomDecoupledHypergraphPlusSGC', 'CustomDecoupledCliqueExpandSGC']:
        PDGNN = {
            'CustomDecoupledSGC':               CustomDecoupledSGC,
            'CustomDecoupledS2GC':              CustomDecoupledS2GC,
            'CustomDecoupledAPPNP':             CustomDecoupledAPPNP,
            'CustomFDGNN':                      CustomFDGNN,
            'CustomDecoupledHypergraphSGC':     CustomDecoupledHypergraphSGC,
            'CustomDecoupledHypergraphPlusSGC': CustomDecoupledHypergraphPlusSGC,
            'CustomDecoupledCliqueExpandSGC':   CustomDecoupledCliqueExpandSGC,
        }
        model = PDGNN[args.backbone](args)
    if getattr(args, 'p1a_encoder', None):
        if args.p1a_encoder == 'graph4_noloop_norm':
            from training.graph4_noloop_norm import Graph4NoLoopNorm
            model.neighbor_agg = Graph4NoLoopNorm()
            print('[GRAPH4_NOLOOP_NORM] (R A0 R)^4 X; remove all self-loops; '
                  'R=clamp(loop-free in-degree, min=1)^-1/2; original direction/multiplicity')
        elif args.p1a_encoder == 'graph4_noloop_sum':
            from training.graph4_noloop_sum import Graph4NoLoopSum
            model.neighbor_agg = Graph4NoLoopSum()
            print('[GRAPH4_NOLOOP_SUM] A0^4 X; remove all self-loops; '
                  'no degree normalization; original direction and non-self edge multiplicity')
        elif args.p1a_encoder == 'hgnn_sym2':
            from .standard_hgnn import FixedHGNNEncoder
            model.neighbor_agg = FixedHGNNEncoder(k=2)
            print('[HGNN_PROP] P^2 X; binary symmetrized complete ego incidence; '
                  'W=I; separate Dv/De; no truncation; fixed-vector replay')
        elif args.p1a_encoder in ('gclean4', 'grow4', 'gbi4'):
            from training.hse_graph_controls import GraphControlEncoder
            model.neighbor_agg = GraphControlEncoder(args.p1a_encoder)
        elif args.p1a_encoder in ('gmean2', 'gattn2'):
            from training.hse_group_trial import GroupTrialEncoder
            model.neighbor_agg = GroupTrialEncoder(args.p1a_encoder)
        else:
            from training.p1a_encoders import P1AEncoder
            model.neighbor_agg = P1AEncoder(args.p1a_encoder)
    return model
