import traceback
import argparse
from training.results import show_final_APAF
from training.utils import set_seed, remove_illegal_characters,str2dict,compose_hyper_params,assign_hyp_param
from distutils.util import strtobool
from pipeline import *
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HGR-CL (internal method key: TEM)')
    parser.add_argument("--dataset", type=str, default='CoraFull-CL', help='Products-CL, Reddit-CL, Arxiv-CL, CoraFull-CL')
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to use.")
    parser.add_argument("--seed", type=int, default=1, help="seed for exp")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs, default = 200")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4, help="weight decay")
    parser.add_argument('--backbone', type=str, default='CustomDecoupledHypergraphSGC', help="HGR-CL fixed hypergraph structural encoder")
    parser.add_argument('--method', type=str,
                        choices=["TEM"], default="TEM",
                        help="HGR-CL continual learning method")
    # parameters for continual learning settings
    parser.add_argument('--share-labels', type=strtobool, default=False,
                        help='task-IL specific, whether to share output label space for different tasks')
    parser.add_argument('--inter-task-edges', type=strtobool, default=False,
                        help='whether to keep the edges connecting nodes from different tasks')
    parser.add_argument('--classifier-increase', type=strtobool, default=True,
                        help='(deprecated) class-IL specific, whether to enlarge the label space with the coming of new classes, unrealistic to be set as False')

    # extra parameters
    parser.add_argument('--refresh_data', type=strtobool, default=False, help='whether to load existing splitting or regenerate')
    parser.add_argument('--d_dtat', default=None, help='will be assigned during running')
    parser.add_argument('--n_cls', default=None, help='will be assigned during running')
    parser.add_argument('--ratio_valid_test', nargs='+', default=[0.2, 0.2], help='ratio of nodes used for valid and test')
    parser.add_argument('--transductive', type=strtobool, default=True, help='using transductive or inductive')
    parser.add_argument('--default_split', type=strtobool, default=False, help='whether to  use the data split provided by the dataset')
    parser.add_argument('--task_seq', default=[])
    parser.add_argument('--n-task', default=0, help='will be assigned during running')
    parser.add_argument('--n_cls_per_task', type=int, default=2, help='how many classes does each task  contain')
    parser.add_argument('--CustomDecoupledS2GC_args',
                        default={'h_dims': [256], 'dropout': 0.0, 'bias': False, 'k': 1, 'alpha': 0.05,
                                 'batch_norm': False, 'linear_bias': False, 'linear': 'nn.Linear'})
    parser.add_argument('--CustomDecoupledAPPNP_args',
                        default={'h_dims': [256], 'dropout': 0.0, 'bias': False, 'k': 2, 'alpha': 0.05,
                                 'batch_norm': False, 'linear_bias': False, 'linear': 'nn.Linear'})
    parser.add_argument('--CustomDecoupledSGC_args',
                        default={'h_dims': [256], 'dropout': 0.0, 'bias': False, 'k': 2, 'alpha': 0.05,
                                 'batch_norm': False, 'linear_bias': False, 'linear': 'nn.Linear'})
    parser.add_argument('--CustomDecoupledHypergraphSGC_args',
                        default={'h_dims': [256], 'dropout': 0.0, 'bias': False, 'k': 2,
                                 'batch_norm': False, 'linear_bias': False, 'linear': 'nn.Linear',
                                 'max_hyperedge_size': 32})
    parser.add_argument('--GAT-args',
                        default={'num_layers': 1, 'num_hidden': 32, 'heads': 8, 'out_heads': 1, 'feat_drop': .6,
                                 'attn_drop': .6, 'negative_slope': 0.2, 'residual': False})
    parser.add_argument('--GCN-args', default={'h_dims': [256], 'dropout': 0.0, 'batch_norm': False})
    parser.add_argument('--GIN-args', default={'h_dims': [256], 'dropout': 0.0})
    parser.add_argument('--SGC_args', default={'h_dims': [256], 'dropout': 0.0, 'bias': False, 'k': 2, 'alpha': 0.05, 'batch_norm': False, 'linear_bias': False, 'linear': 'nn.Linear'})
    parser.add_argument('--tem_args', type=str2dict, default={
        'budget': 400, 'sampler': 'hyperedge_aware', 'lambda_sc': 0.5
    })
    parser.add_argument('--cls-balance', type=strtobool, default=True, help='whether to balance the cls when training and testing')
    parser.add_argument('--repeats', type=int, default=5, help='how many times to repeat the experiments for the mean and std')
    parser.add_argument('--memory_audit', type=strtobool, default=False,
                        help='Log TEM full-batch memory payload and run provenance to JSONL')
    parser.add_argument('--ILmode', default='classIL',choices=['taskIL','classIL'])
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--minibatch', type=strtobool, default=False, help='whether to use the mini-batch training')
    parser.add_argument('--batch_shuffle', type=strtobool, default=False, help='whether to shuffle the data when constructing the dataloader')
    parser.add_argument('--sample_nbs', type=strtobool, default=True, help='whether to sample neighbors instead of using all')
    parser.add_argument('--n_nbs_sample', type=lambda x: [int(i) for i in x.replace(' ', '').split(',')], default=[10, 20], help='number of neighbors to sample per hop, use comma to separate the numbers when using the command line, e.g. 10,25 or 10, 25')
    parser.add_argument('--nb_sampler', default=None)
    parser.add_argument('--replace_illegal_char', type=strtobool, default=False)
    parser.add_argument('--ori_data_path', type=str, default='./raw_data', help='the root path to raw data')
    parser.add_argument('--data_path', type=str, default='./data', help='the path to processed data (splitted into tasks)')
    parser.add_argument('--result_path', type=str, default='./results', help='the path for saving results')
    parser.add_argument('--overwrite_result', type=strtobool, default=False, help='whether to overwrite existing results')
    parser.add_argument('--load_check', type=strtobool, default=False, help='whether to check the existence of processed data by loading')
    parser.add_argument('--perform_testing', type=strtobool, default=True, help='whether to check the existence of processed data by loading')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='experiment subfolder name; if set, all outputs go to {result_path}/{exp_name}/')
    parser.add_argument('--use_hypergraph_expansion', type=strtobool, default=False,
                        help='[doc] whether the data_path points to hypergraph-expanded data')
    parser.add_argument('--hyperedge_construction', type=str, default='onehop',
                        choices=['onehop'],
                        help='[doc] hyperedge construction strategy of externally prepared data (not generated by this flag)')
    parser.add_argument('--hypergraph_mode', type=str, default='augment',
                        choices=['augment', 'replace'],
                        help='[doc] graph expansion mode of externally prepared data (not generated by this flag)')
    parser.add_argument('--max_hyperedge_size', type=int, default=None,
                        help='[doc] max nodes per hyperedge of externally prepared data (not generated by this flag)')

    parser.add_argument('--hgnn_k', type=int, default=None,
                        help='Override k for CustomDecoupledHypergraphSGC (default: use CustomDecoupledHypergraphSGC_args)')
    parser.add_argument('--hgnn_max_he_size', type=int, default=None,
                        help='Override max_hyperedge_size for CustomDecoupledHypergraphSGC (default: use CustomDecoupledHypergraphSGC_args)')

    parser.add_argument('--p1a_encoder', choices=['raw', 'graph1', 'graph2', 'graph4', 'hse2', 'gmean2', 'gattn2', 'gclean4', 'grow4', 'gbi4', 'hgnn_sym2', 'graph4_noloop_sum', 'graph4_noloop_norm'],
                        default=None, help='Opt-in P1-A encoder; keeps the original TEM pipeline and repeats')
    args = parser.parse_args()
    args.tem_args.setdefault('lambda_sc', 0.5)
    if 'p1e_mode' in args.tem_args:
        from training.p1e_routing import MODES
        if args.tem_args['p1e_mode'] not in MODES:
            parser.error('p1e_mode must be none, edge, memory, or switch')
        if (args.method != 'TEM' or args.backbone != 'CustomDecoupledHypergraphSGC'
                or args.minibatch or args.ILmode != 'classIL' or args.inter_task_edges):
            parser.error('p1e_mode requires full-batch TEM, HSE backbone, classIL, no inter-task edges')
    if args.p1a_encoder and (args.method != 'TEM' or args.backbone != 'CustomDecoupledHypergraphSGC'
                             or args.minibatch or args.ILmode != 'classIL' or args.inter_task_edges):
        parser.error('--p1a_encoder requires full-batch TEM, CustomDecoupledHypergraphSGC, classIL, no inter-task edges')
    if args.p1a_encoder and args.hgnn_k not in (None, 2):
        parser.error('--p1a_encoder defines propagation depth; do not combine with a different --hgnn_k')
    if args.p1a_encoder == 'graph4_noloop_norm':
        args.graph4_propagation_config = dict(steps=4, self_loops='remove_all',
            normalization='symmetric_in_degree_clamp1', aggregation='normalized_sum',
            degree_graph='loop_free', direction='input',
            nonself_multiplicity='preserved', learned_encoder=False)
    if args.p1a_encoder == 'graph4_noloop_sum':
        args.graph4_propagation_config = dict(steps=4, self_loops='remove_all',
            normalization='none', aggregation='sum', direction='input',
            nonself_multiplicity='preserved', learned_encoder=False)
    if args.p1a_encoder == 'hgnn_sym2':
        if args.hgnn_max_he_size is not None or args.max_hyperedge_size is not None or args.use_hypergraph_expansion:
            parser.error('hgnn_sym2 uses complete ego incidence without truncation; use original task graphs')
        args.hgnn_propagation_config = dict(operator='Dv^-1/2 H W De^-1 H.T Dv^-1/2',
            k=2, construction='symmetrized_binary_ego', edge_weights='unit',
            max_hyperedge_size=None, learned_encoder=False)
    if args.memory_audit and (args.method != 'TEM' or args.minibatch):
        parser.error('--memory_audit currently supports TEM full-batch training only')
    args.ratio_valid_test = [float(i) for i in args.ratio_valid_test]

    if args.hgnn_k is not None:
        args.CustomDecoupledHypergraphSGC_args['k'] = args.hgnn_k
    if args.hgnn_max_he_size is not None:
        args.CustomDecoupledHypergraphSGC_args['max_hyperedge_size'] = args.hgnn_max_he_size


    if args.exp_name:
        args.result_path = os.path.join(args.result_path, args.exp_name)
    mkdir_if_missing(args.result_path)

    set_seed(args)
    mkdir_if_missing(f'{args.data_path}')

    method_args = {'TEM': args.tem_args}
    backbone_args = {
                     'CustomDecoupledSGC': {'k': args.CustomDecoupledSGC_args['k']},
                     'CustomDecoupledS2GC': args.CustomDecoupledSGC_args,
                     'CustomDecoupledAPPNP': args.CustomDecoupledAPPNP_args, 'GCN': args.GCN_args, 'GAT': args.GAT_args, 'GIN': args.GIN_args, 'SGC': args.SGC_args,
                     'CustomDecoupledHypergraphSGC': {'k': args.CustomDecoupledHypergraphSGC_args['k'], 'max_he': args.CustomDecoupledHypergraphSGC_args['max_hyperedge_size']},
                     'CustomDecoupledHypergraphPlusSGC': {'k': args.CustomDecoupledHypergraphSGC_args['k'], 'max_he': args.CustomDecoupledHypergraphSGC_args['max_hyperedge_size']},
                     }
    hyp_param_list = compose_hyper_params(method_args[args.method])
    AP_best, name_best, hyp_best_str = 0, None, None
    AP_dict = {str(hyp_params).replace("'",'').replace(' ','').replace(',','_').replace(':','_'):[] for hyp_params in hyp_param_list}
    AF_dict = {str(hyp_params).replace("'",'').replace(' ','').replace(',','_').replace(':','_'):[] for hyp_params in hyp_param_list}
    PM_dict = {str(hyp_params).replace("'",'').replace(' ','').replace(',','_').replace(':','_'):[] for hyp_params in hyp_param_list}
    for hyp_params in hyp_param_list:
        # iterate over each candidate hyper-parameter combination, and find the best one over the validation set
        hyp_params_str = str(hyp_params).replace("'",'').replace(' ','').replace(',','_').replace(':','_')
        print(hyp_params_str)
        assign_hyp_param(args,hyp_params)
        args.nb_sampler = dgl.dataloading.MultiLayerNeighborSampler(args.n_nbs_sample) if args.sample_nbs else dgl.dataloading.MultiLayerFullNeighborSampler(2)
        main = get_pipeline(args)
        train_ratio = round(1 - args.ratio_valid_test[0] - args.ratio_valid_test[1], 2)
        if args.ILmode == 'classIL':
            subfolder = f'inter_task_edges/cls_IL/train_ratio_{train_ratio}/' if args.inter_task_edges else f'no_inter_task_edges/cls_IL/train_ratio_{train_ratio}/'
        elif args.ILmode == 'taskIL':
            subfolder = f'inter_task_edges/tsk_IL/train_ratio_{train_ratio}/' if args.inter_task_edges else f'no_inter_task_edges/tsk_IL/train_ratio_{train_ratio}/'

        if args.p1a_encoder:
            backbone_args[args.backbone]['p1a_encoder'] = args.p1a_encoder
            if args.p1a_encoder == 'hgnn_sym2':
                backbone_args[args.backbone]['max_he'] = None
        name = f'{subfolder}val_{args.dataset}_{args.n_cls_per_task}_{args.method}_{list(hyp_params.values())}_{args.backbone}_{backbone_args[args.backbone]}_{args.classifier_increase}_{args.cls_balance}_{args.epochs}_{args.repeats}'
        if args.minibatch:
            name = name + f'_bs{args.batch_size}'
        mkdir_if_missing(f'{args.result_path}/' + subfolder)
        if args.replace_illegal_char:
            name = remove_illegal_characters(name)

        if os.path.isfile(f'{args.result_path}/{name}.pkl') and not args.overwrite_result:
            # if the results exist, directly use them without over-writting
            print('The results of the following configuration exists, will load this result for use \n', f'{args.result_path}/{name}.pkl')
            AP_AF = show_final_APAF(f'{args.result_path}/{name}.pkl')
            AP = float(AP_AF.split('$')[0])
            if AP > AP_best:
                AP_best = AP
                hyp_best_str = hyp_params_str
                name_best = name
                print(f'best params is {hyp_best_str}, best AP is {AP_best}')
            continue
        else:
            # if results do not exist or choose to overwrite existing results
            acc_matrices = []
            print('method args are', hyp_params)
            for ite in range(args.repeats):
                print(name, ite)
                args.current_model_save_path = [name,ite]
                try:
                    AP, AF, acc_matrix = main(args,valid=True)
                    AP_dict[hyp_params_str].append(AP)
                    AF_dict[hyp_params_str].append(AF)

                    # choose the best configuration according to the validation results
                    acc_matrices.append(acc_matrix)
                    torch.cuda.empty_cache()
                    with open(f'{args.result_path}/experiment_log.txt', 'a') as f:
                        f.write(f'[VAL] {name} | repeat={ite} | AP={AP:.4f} | AF={AF:.4f}\n')
                except Exception as e:
                    mkdir_if_missing(f'{args.result_path}/errors/' + subfolder)
                    if ite > 0:
                        name_ = f'{subfolder}val_{args.dataset}_{args.n_cls_per_task}_{args.method}_{list(hyp_params.values())}_{args.backbone}_{backbone_args[args.backbone]}_{args.classifier_increase}_{args.cls_balance}_{args.epochs}_{ite}'
                        with open(f'{args.result_path}/{name_}.pkl', 'wb') as f:
                            pickle.dump(acc_matrices, f)
                    print('error', e)
                    name = 'errors/{}'.format(name)
                    acc_matrices = traceback.format_exc()
                    print(acc_matrices)
                    print('error happens on \n', name)
                    torch.cuda.empty_cache()
                    break
            ap_val = AP_dict[hyp_params_str]
            af_val = AF_dict[hyp_params_str]
            if ap_val:
                with open(f'{args.result_path}/experiment_log.txt', 'a') as f:
                    f.write(f'[VAL] {name} | MEAN | AP={np.mean(ap_val):.4f}±{np.std(ap_val):.4f} | AF={np.mean(af_val):.4f}±{np.std(af_val):.4f}\n')
            if np.mean(AP_dict[hyp_params_str]) > AP_best:
                AP_best = np.mean(AP_dict[hyp_params_str])
                hyp_best_str = hyp_params_str
                name_best = name
            print(f'best params is {hyp_best_str}, best AP is {AP_best}')
            with open(f'{args.result_path}/{name}.pkl', 'wb') as f:
                pickle.dump(acc_matrices, f)

    # save the models
    if name_best is None:
        print('[ERROR] All experiment repeats failed. Check error messages above.')
        sys.exit(1)
    config_name = name_best.split('/')[-1]
    subfolder_c = name_best.split(config_name)[-2]
    if args.perform_testing:
        print('----------Now in testing--------')
        acc_matrices = []
        ap_test_list, af_test_list = [], []
        for ite in range(args.repeats):
            args.current_model_save_path = [name_best, ite]
            AP_test, AF_test, acc_matrix_test = main(args, valid=False)
            acc_matrices.append(acc_matrix_test)
            ap_test_list.append(AP_test)
            af_test_list.append(AF_test)
            with open(f'{args.result_path}/experiment_log.txt', 'a') as f:
                f.write(f'[TEST] {name_best} | repeat={ite} | AP={AP_test:.4f} | AF={AF_test:.4f}\n')
        with open(f'{args.result_path}/{name_best}.pkl'.replace('val', 'te'), 'wb') as f:
            pickle.dump(acc_matrices, f)
        ap_arr = np.array(ap_test_list)
        af_arr = np.array(af_test_list)
        with open(f'{args.result_path}/experiment_log.txt', 'a') as f:
            f.write(f'[TEST] {name_best} | MEAN | AP={ap_arr.mean():.4f}±{ap_arr.std():.4f} | AF={af_arr.mean():.4f}±{af_arr.std():.4f}\n\n')

