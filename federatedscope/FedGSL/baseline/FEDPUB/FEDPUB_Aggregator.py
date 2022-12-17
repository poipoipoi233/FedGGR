import os
from collections import OrderedDict

import torch
from federatedscope.core.aggregators import Aggregator
from federatedscope.core.auxiliaries.utils import param2tensor
import numpy as np
from scipy.spatial.distance import cosine


class FEDPUB_Aggregator(Aggregator):
    """Implementation of vanilla FedAvg refer to `Communication-efficient
    learning of deep networks from decentralized data` [McMahan et al., 2017]
        (http://proceedings.mlr.press/v54/mcmahan17a.html)
    """

    def __init__(self, model=None, device='cpu', config=None):
        super(Aggregator, self).__init__()
        self.model = model
        self.device = device
        self.cfg = config

    def aggregate(self, agg_info):
        """
        To preform aggregation

        Arguments:
        agg_info (dict): the feedbacks from clients
        :returns: the aggregated results
        :rtype: dict
        """

        models = agg_info["client_feedback"]
        recover_fun = agg_info['recover_fun'] if (
                'recover_fun' in agg_info and self.cfg.federate.use_ss) else None
        avg_model = self._para_weighted_avg(models, recover_fun=recover_fun)

        return avg_model

    def update(self, model_parameters):
        '''
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        '''
        self.model.load_state_dict(model_parameters, strict=False)

    def save_model(self, path, cur_round=-1):
        assert self.model is not None

        ckpt = {'cur_round': cur_round, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

    def load_model(self, path):
        assert self.model is not None

        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            return ckpt['cur_round']
        else:
            raise ValueError("The file {} does NOT exist".format(path))

    def _para_weighted_avg(self, models, recover_fun=None):
        # TODO: 确保传进来的models是按client编号升序排列的 （ √）重要
        # 参考FED-PUB源码 server.py文件的update函数
        training_set_size = 0
        local_weights = []
        local_proxy_outputs = []
        local_train_sizes = []
        aggr_local_model_weights = dict()
        for i in range(len(models)):
            sample_size, weight, proxy_output = models[i]
            local_weights.append(weight)
            local_proxy_outputs.append(proxy_output)
            local_train_sizes.append(sample_size)

        n_connected = len(models)  # TODO: 考虑客户端采样情况，乘以self.args.frac
        assert n_connected == len(local_proxy_outputs)
        sim_matrix = np.empty(shape=(n_connected, n_connected))

        for i in range(n_connected):  # 相似度矩阵计算
            for j in range(n_connected):
                similarity = 1 - cosine(
                    local_proxy_outputs[i], local_proxy_outputs[j]
                )
                sim_matrix[i, j] = similarity

        if self.cfg.model.fedpub.agg_norm == 'exp':
            sim_matrix = np.exp(self.cfg.model.fedpub.norm_scale * sim_matrix)

        if self.cfg.model.fedpub.cluster:
            for i in range(n_connected):
                mask = (sim_matrix[i] < sim_matrix[i].mean())
                sim_matrix[i][mask] = 0

        row_sums = sim_matrix.sum(axis=1)
        sim_matrix = sim_matrix / row_sums[:, np.newaxis]

        for i in range(len(models)):
            local_ratio = sim_matrix[i, :]
            aggr_local_model_weights[i+1]=self.fedpub_aggregate(local_weights, local_ratio)

        return aggr_local_model_weights

    def get_active(self, mask):
        active = np.absolute(mask) >= self.cfg.model.fedpub.l1
        return active.astype(float)

    def fedpub_aggregate(self, local_weights, ratio=None):
        # 参考FED-PUB aggregate.py中的aggregate函数
        aggr_theta = OrderedDict([(k, None) for k in local_weights[0].keys()])
        if ratio is not None:
            for name, params in aggr_theta.items():
                if self.cfg.model.fedpub.mask_aggr:  # 一直为False
                    if 'mask' in name:
                        # get active
                        acti = [ratio[i] * self.get_active(lw[name]) + 1e-8 for i, lw in enumerate(local_weights)]
                        # get element_wise ratio
                        elem_wise_ratio = acti / np.sum(acti, 0)
                        # perform element_wise aggr
                        aggr_theta[name] = np.sum(
                            [theta[name] * elem_wise_ratio[j] for j, theta in enumerate(local_weights)], 0)
                    else:
                        aggr_theta[name] = np.sum([theta[name] * ratio[j] for j, theta in enumerate(local_weights)], 0)
                else:
                    aggr_theta[name] = np.sum([theta[name] * ratio[j] for j, theta in enumerate(local_weights)], 0)
        else:
            ratio = 1 / len(local_weights)
            for name, params in aggr_theta.items():
                aggr_theta[name] = np.sum([theta[name] * ratio for j, theta in enumerate(local_weights)], 0)
            # self.logger.print(f'weight aggregation done ({round(time.time()-st, 3)} s)')
        return aggr_theta

    def convert_np_to_tensor(self, state_dict, gpu_id, skip_stat=False, skip_mask=False, model=None):
        # 源代码来自FED-PUB utils.py文件中的convert_np_to_tensor函数
        _state_dict = OrderedDict()
        for k, v in state_dict.items():
            if skip_stat:
                if 'running' in k or 'tracked' in k:
                    _state_dict[k] = model[k]
                    continue
            if skip_mask:
                if 'mask' in k or 'pre' in k or 'pos' in k:
                    _state_dict[k] = model[k]
                    continue

            if len(np.shape(v)) == 0:
                _state_dict[k] = torch.tensor(v).cuda(gpu_id)
            else:
                _state_dict[k] = torch.tensor(v).requires_grad_().cuda(gpu_id)
        return _state_dict
