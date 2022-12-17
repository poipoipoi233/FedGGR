import torch
from federatedscope.register import register_trainer
from federatedscope.gfl.trainer.nodetrainer import NodeFullBatchTrainer
from federatedscope.core.auxiliaries.enums import LIFECYCLE
from federatedscope.core.trainers.context import CtxVar

class FedGSL_Trainer(NodeFullBatchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FedGSL_Trainer, self).__init__(model, data, device, config,
                                             only_for_eval, monitor)

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        batch_x, edge_index, glob_emb = batch.x, batch.edge_index, ctx.glob_emb.to(ctx.device)
        pred = ctx.model(batch_x=batch_x, edge_index=edge_index, glob_emb=glob_emb)[
            batch['{}_mask'.format(ctx.cur_split)]]

        label = batch.y[batch['{}_mask'.format(ctx.cur_split)]]
        ctx.batch_size = torch.sum(ctx.data_batch['{}_mask'.format(
            ctx.cur_split)]).item()

        ctx.loss_batch = CtxVar(ctx.criterion(pred, label), LIFECYCLE.BATCH)
        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)

def call_my_torch_trainer(trainer_type):
    if trainer_type == 'fedgsl_trainer':
        trainer_builder = FedGSL_Trainer
        return trainer_builder


register_trainer('fedgsl_trainer', call_my_torch_trainer)
