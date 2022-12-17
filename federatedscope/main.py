import os
import sys

DEV_MODE = True  # simplify the federatedscope re-setup everytime we change
# the source codes of federatedscope
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, \
    get_server_cls
from federatedscope.core.configs.config import global_cfg, CfgNode
from federatedscope.core.fed_runner import FedRunner

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

if __name__ == '__main__':
    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # load clients' cfg file
    if args.client_cfg_file:
        client_cfgs = CfgNode.load_cfg(open(args.client_cfg_file, 'r'))
        # client_cfgs.set_new_allowed(True)
        client_cfgs.merge_from_list(client_cfg_opt)
    else:
        client_cfgs = None

    # federated dataset might change the number of clients
    # thus, we allow the creation procedure of dataset to modify the global
    # cfg object
    data, modified_cfg = get_data(config=init_cfg.clone(),
                                  client_cfgs=client_cfgs)
    init_cfg.merge_from_other_cfg(modified_cfg)

    init_cfg.freeze()

    runner = FedRunner(data=data,
                       server_class=get_server_cls(init_cfg),
                       client_class=get_client_cls(init_cfg),
                       config=init_cfg.clone(),
                       client_configs=client_cfgs)
    _ = runner.run()
    import pandas as pd

    out_dict = {
        'seed': [init_cfg.seed],
        'method': [init_cfg.federate.method],
        'model': [init_cfg.model.type],
        'batch_size': [init_cfg.dataloader.batch_size],
        'datasets': [init_cfg.data.type],
        'client_num': [init_cfg.federate.client_num],
        'hidden': [init_cfg.model.hidden],
        'dropout': [init_cfg.model.dropout],
        'local_updates': [init_cfg.train.local_update_steps],
        'lr': [init_cfg.train.optimizer.lr],
        'weight_decay': [init_cfg.train.optimizer.weight_decay],
        'test_acc': [_['client_summarized_avg']['test_acc']],
        'val_loss': [_['client_summarized_avg']['val_loss']]
    }
    if out_dict['method'][0] == 'fedgsl':
        out_dict['server_lr'] = init_cfg.model.fedgsl.server_lr
        out_dict['loc_gnn_outsize'] = init_cfg.model.fedgsl.loc_gnn_outsize
        out_dict['glob_gnn_outsize'] = init_cfg.model.fedgsl.glob_gnn_outsize
        out_dict['gsl_gnn_hids'] = init_cfg.model.fedgsl.gsl_gnn_hids
        out_dict['k_for_knn'] = init_cfg.model.fedgsl.k
        out_dict['pretrain_out_channels']= init_cfg.model.fedgsl.pretrain_out_channels

    df = pd.DataFrame(out_dict, columns=out_dict.keys())
    # folder_name = f'federatedscope/FedGSL/exp_out/{init_cfg.federate.method}_{init_cfg.model.type}/{init_cfg.data.type}'
    # parameter_name = f'{init_cfg.federate.client_num}clients_{init_cfg.train.optimizer.type}'
    #
    # if init_cfg.model.fedgsl.HPO is True:
    #     folder_name = folder_name + '_HPO'
    #     parameter_name = parameter_name + '_HPO'
    # else:
    #     folder_name = folder_name + '_exp'
    #     parameter_name = parameter_name + '_exp'
    #
    # csv_name = f'{folder_name}/{parameter_name}.csv'
    #
    # if not os.path.exists(folder_name):
    #     os.makedirs(folder_name)

    df.to_csv(f'{init_cfg.data.type}_sensitivity.csv', mode='a', index=False, header=False)
