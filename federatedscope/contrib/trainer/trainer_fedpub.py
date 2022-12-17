import inspect
import torch

from federatedscope.register import register_trainer
from federatedscope.core.trainers import BaseTrainer
from federatedscope.gfl.trainer.nodetrainer import NodeFullBatchTrainer
from federatedscope.core.auxiliaries.enums import LIFECYCLE
from federatedscope.core.monitors import Monitor
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.auxiliaries.utils import param2tensor, \
    merge_param_dict

# An example for converting torch training process to FS training process

# Refer to `federatedscope.core.trainers.BaseTrainer` for interface.

class FEDPUB_trainer(NodeFullBatchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FEDPUB_trainer, self).__init__(model, data, device, config,
                                             only_for_eval, monitor)
        ############################################################################
        self.model = model
        self.device=device
        self.masks = []
        self.mask_rank = config.model.fedpub.mask_rank
        self.l1 = config.model.fedpub.l1
        self.loc_l2 = config.model.fedpub.loc_l2
        for name, param in model.state_dict().items():
            if 'mask' in name and self.mask_rank == -1:
                self.masks.append(param)

        # measuring sparsity per epoch for this case is not working
        if self.mask_rank != -1:
            for module in self.model.children():
                self.masks.append(module.mask)
        #############################################################################

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        pred = ctx.model(batch)[batch['{}_mask'.format(ctx.cur_split)]]
        label = batch.y[batch['{}_mask'.format(ctx.cur_split)]]
        ctx.batch_size = torch.sum(ctx.data_batch['{}_mask'.format(
            ctx.cur_split)]).item()
        loss_clf = ctx.criterion(pred, label)
        loss_sparsity = 0.0
        loss_proximal_term = 0.0

        #############################################################################################
        for name, param in self.model.state_dict().items():
            if 'mask' in name and self.mask_rank == -1:
                loss_sparsity += torch.norm(param.float(), 1) * self.l1  # 损失函数的第二项
            elif 'conv' in name or 'clsif' in name:
                if self.ctx.curr_rnd > 0:
                    loss_proximal_term += torch.norm(param.float() - self.ctx.W_old[name].cuda(self.device), 2) * self.loc_l2

        if self.mask_rank != -1:
            for module in self.model.children():
                loss_sparsity += torch.norm(module.mask.float(), 1) * self.l1

        loos_batch = loss_clf + loss_sparsity + loss_proximal_term
        #################################################################################################

        ctx.loss_batch = CtxVar(loos_batch, LIFECYCLE.BATCH)
        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)



    @torch.no_grad()
    def get_proxy_output(self):
        model=self.ctx.model.to(self.ctx.device)
        model.eval()
        proxy_in = self.ctx.proxy_data
        proxy_in = proxy_in.cuda(self.device)
        proxy_out = model(proxy_in, is_proxy=True)
        proxy_out = proxy_out.mean(dim=0)
        proxy_out = proxy_out.clone().detach().cpu().numpy()
        return proxy_out


def call_my_torch_trainer(trainer_type):
    if trainer_type == 'fedpub_trainer':
        trainer_builder = FEDPUB_trainer
        return trainer_builder


register_trainer('fedpub_trainer', call_my_torch_trainer)
