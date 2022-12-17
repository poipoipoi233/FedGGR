from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_FEDPUB_cfg(cfg):
    '''
    Define configuration parameters for FEDPUB
    Source: Supplementary material of FEDPUB in the webpage 'https://openreview.net/forum?id=ToYi8C6fetv'
    '''
    cfg.model.fedpub = CN()
    cfg.model.fedpub.task = "Cora_CC_total_0.2_HET"
    cfg.model.fedpub.backbone = 'GNN'
    cfg.model.fedpub.laye_mask_one = False
    cfg.model.fedpub.mask_rank = -1
    cfg.model.fedpub.mask_drop = False
    cfg.model.fedpub.mask_drop_ratio = 0.5
    cfg.model.fedpub.mask_noise = False
    cfg.model.fedpub.no_clsf_mask = False
    cfg.model.fedpub.clsf_mask_one = False

    cfg.model.fedpub.n_feat = 1433
    cfg.model.fedpub.n_proxy = 5

    cfg.model.fedpub.l1 = 1e-3
    cfg.model.fedpub.loc_l2 = 1e-3
    cfg.model.fedpub.agg_norm = "exp"
    cfg.model.fedpub.cluster = True
    cfg.model.fedpub.mask_aggr = False
    cfg.model.fedpub.norm_scale = 10
    cfg.register_cfg_check_fun(assert_FEDPUB_cfg)


def assert_FEDPUB_cfg(cfg):
    pass


register_config("fedpub", extend_FEDPUB_cfg)
