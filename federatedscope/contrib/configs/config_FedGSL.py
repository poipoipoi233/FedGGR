from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_FedGSL_cfg(cfg):
    # ------------------------------------------------------------------------ #
    # FedGSL related options
    # ------------------------------------------------------------------------ #
    cfg.model.fedgsl = CN()
    cfg.fedgsl = CN()

    # -----------------------------------model --------------------------------#
    # loc_pretrain for Auto-Encoder
    cfg.model.fedgsl.pretrain_out_channels = 64
    cfg.model.fedgsl.pretrain_epoch = 200
    cfg.model.fedgsl.AE_lr = 0.2
    cfg.model.fedgsl.update_RawNodeFeature = False

    # for the local graph learning module
    cfg.model.fedgsl.gnn_layer = 2
    cfg.model.fedgsl.loc_mlp_layer = 1
    cfg.model.fedgsl.loc_mlp_hids = 64
    cfg.model.fedgsl.gpr_layer = 10
    cfg.model.fedgsl.loc_gnn_hid = 64
    cfg.model.fedgsl.loc_gnn_outsize = 256

    # for the global graph structure learning module
    cfg.model.fedgsl.glob_gnn_outsize = 64
    cfg.model.fedgsl.node_dim = 64
    cfg.model.fedgsl.gsl_gnn_hids = 64
    cfg.model.fedgsl.k = 10
    cfg.model.fedgsl.server_lr = 0.001
    cfg.model.fedgsl.gsl_mlp_layer = 2
    cfg.model.fedgsl.gsl_gnn_dropout = 0.5
    cfg.model.fedgsl.gsl_adj_dropout = 0
    cfg.model.fedgsl.gsl_optimizer_type = 'Adam'
    cfg.model.fedgsl.gsl_clip_grad = False
    cfg.model.fedgsl.generator = 'MY-MLP'

    # cfg.model.fedgsl.gsl_graph_gen_osize = 64
    #HPO option for output
    cfg.model.fedgsl.HPO = False
    # -----------------------Local Differential Privacy (LDP)-------------------#
    cfg.model.fedgsl.add_noise = False
    cfg.model.fedgsl.LDP_lambda = 0.015

    # --------------- register corresponding check function ---------------------#
    cfg.register_cfg_check_fun(assert_training_cfg)


def assert_training_cfg(cfg):
    pass


register_config("fedst", extend_FedGSL_cfg)
