import torch
import logging
import copy
import numpy as np
import pickle
import sys
from federatedscope.core.message import Message
from federatedscope.core.workers.server import Server
from federatedscope.FedGSL.model.server_models import *
from federatedscope.core.auxiliaries.sampler_builder import get_sampler
from federatedscope.core.workers.client import Client
from federatedscope.core.auxiliaries.utils import merge_dict, Timeout, \
    merge_param_dict
import torch.optim as optim
from federatedscope.core.auxiliaries.utils import merge_dict
from federatedscope.FedGSL.model.AE_models import localAE
from federatedscope.register import register_worker
import pandas as pd

logger = logging.getLogger(__name__)


class FedGSL_Server(Server):
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cuda:0',
                 strategy=None,
                 **kwargs):
        super(FedGSL_Server,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)

        # TODO 把global_dropout 写进yaml文件 ()
        # fedgsl_cfg = self._cfg.model.fedgsl
        # self.global_GSL = GSL(in_channels=fedgsl_cfg.pretrain_out_channels,
        #                       glob_gnn_outsize=fedgsl_cfg.glob_gnn_outsize,
        #                       gsl_gnn_hids=fedgsl_cfg.gsl_gnn_hids,
        #                       dropout_GNN=fedgsl_cfg.gsl_gnn_dropout,
        #                       dropout_adj_rate=fedgsl_cfg.gsl_adj_dropout,
        #                       k_forKNN=fedgsl_cfg.k,
        #                       mlp_layers=fedgsl_cfg.gsl_mlp_layer,
        #                       generator=fedgsl_cfg.generator)
        # self.global_GSL.cuda(device)
        self.client_node_num = {}

    def _register_default_handlers(self):
        self.register_handlers('join_in', self.callback_funcs_for_join_in)
        self.register_handlers('join_in_info', self.callback_funcs_for_join_in)  # TODO: 删除,直接加在这里可能会有BUG0
        self.register_handlers('compressed_local_node_embedding', self.callback_funcs_model_para)
        self.register_handlers('metrics', self.callback_funcs_for_metrics)
        self.register_handlers('update_GNN_and_globalModel', self.callback_funcs_model_para)

    def trigger_for_start(self):
        # Server类 callback_funcs_for_join_in()函数的结尾会调用该方法
        # This method will be called at the end of the Server class callback_funcs_for_join_in() function
        # in federatedscope/core/workers/server.py
        """
        To start the FL course when the expected number of clients have joined
        """
        if self.check_client_join_in():
            if self._cfg.federate.use_ss:
                self.broadcast_client_address()

            # get sampler
            if 'client_resource' in self._cfg.federate.join_in_info:
                client_resource = [
                    self.join_in_info[client_index]['client_resource']
                    for client_index in np.arange(1, self.client_num + 1)
                ]
            else:
                if self._cfg.backend == 'torch':
                    model_size = sys.getsizeof(pickle.dumps(
                        self.model)) / 1024.0 * 8.
                else:
                    # TODO: calculate model size for TF Model
                    model_size = 1.0
                    logger.warning(f'The calculation of model size in backend:'
                                   f'{self._cfg.backend} is not provided.')

                client_resource = [
                    model_size / float(x['communication']) +
                    float(x['computation']) / 1000.
                    for x in self.client_resource_info
                ] if self.client_resource_info is not None else None

            if self.sampler is None:
                self.sampler = get_sampler(
                    sample_strategy=self._cfg.federate.sampler,
                    client_num=self.client_num,
                    client_info=client_resource)

            # change the deadline if the asyn.aggregator is `time up`
            if self._cfg.asyn.use and self._cfg.asyn.aggregator == 'time_up':
                self.deadline_for_cur_round = self.cur_timestamp + \
                                              self._cfg.asyn.time_budget

            ##################################################################################
            logger.info(
                '-----------The server sends a message to each client asking them to start pre-training (Round #{:d})-----------'.
                format(self.state)
            )
            self.comm_manager.send(
                Message(msg_type='local_pretrain',
                        sender=self.ID,
                        receiver=list(self.comm_manager.neighbors.keys()),
                        state=self.state,
                        timestamp=self.cur_timestamp))

            ####################################################################################

    def check_and_move_on(self, check_eval_result=False):
        # TODO Confirm the meaning of "self.state"
        # TODO 控制全局轮数增加的过程
        self.client_IDs = [i for i in range(1, self.client_num + 1)]
        move_on_flag = True
        # Transmit model and embedding to get gradient back
        # 这个if里，接收客户端第一次传给服务器的feat_emb,然后生成邻接矩阵做图卷积，并把对应的emb返回给客户端
        if self.check_buffer(self.state, self.client_num) and self.state == 0:
            # we should wait for all messages
            node_emb_matrix_list = []
            node_emb_matrix_dict = dict()
            node_label_matrix_list = []
            node_label_matrix_dict = dict()
            for sender in self.msg_buffer['train'][self.state]:
                content = self.msg_buffer['train'][self.state][sender]
                node_emb_matrix_dict[sender], node_label_matrix_dict[sender] = content[0], content[1]
                self.client_node_num[sender] = node_emb_matrix_dict[sender].shape[0]  # save the number of node of each client
            for client_id in self.client_IDs:
                node_emb_matrix_list.append(node_emb_matrix_dict[client_id])
                # node_label_matrix_list.append(node_label_matrix_dict[client_id]) #TODO 待删除
            node_emb_all = torch.cat(node_emb_matrix_list, dim=0)
            # node_label_all = torch.cat(node_label_matrix_list, dim=0)
            # torch.save(node_emb_all, 'node_emb_all.pt')  # 可视化用
            # torch.save(node_label_all, 'node_label_all.pt')  # 可视化用

            self.node_emb_all = node_emb_all.to(self.device)
            self.client_sampleNum = self.client_node_num.copy()  # 服务器端保存每个client的节点数

            #TODO 定义图结构学习模型 原先在Server类的__init__函数中定义
            fedgsl_cfg = self._cfg.model.fedgsl
            self.global_GSL = GSL(in_channels=fedgsl_cfg.pretrain_out_channels,
                                  glob_gnn_outsize=fedgsl_cfg.glob_gnn_outsize,
                                  gsl_gnn_hids=fedgsl_cfg.gsl_gnn_hids,
                                  dropout_GNN=fedgsl_cfg.gsl_gnn_dropout,
                                  dropout_adj_rate=fedgsl_cfg.gsl_adj_dropout,
                                  k_forKNN=fedgsl_cfg.k,
                                  mlp_layers=fedgsl_cfg.gsl_mlp_layer,
                                  generator=fedgsl_cfg.generator,
                                  client_sampleNum=list(self.client_sampleNum.values()),
                                  device=self.device)
            self.global_GSL.cuda(self.device)

            logger.info(
                f'\tServer #{self.ID}: 已经Concat了所有client的低维节点嵌入矩阵，开始执行第一次图结构学习.'
            )

            if self._cfg.model.fedgsl.gsl_optimizer_type == 'Adam':
                self.GSL_optimizer = optim.Adam(self.global_GSL.parameters(), lr=self._cfg.model.fedgsl.server_lr,
                                                weight_decay=self._cfg.train.optimizer.weight_decay)
            self.global_GSL.train()
            self.GSL_optimizer.zero_grad()
            # Get global embedding
            self.glob_emb = self.global_GSL(self.node_emb_all)
            logger.info(
                f'\tServer #{self.ID}: 得到动态图卷积结果，开始将embedding回传给client@{self.state // 2}.'
            )

            # Server send global embedding matrix to the corresponding client.
            temp = 0
            for receiver_ID in self.client_IDs:
                content = self.glob_emb[temp:temp + self.client_sampleNum[receiver_ID]].detach()  # 选取每个client对应的样本数
                temp += self.client_sampleNum[receiver_ID]
                self.comm_manager.send(
                    Message(msg_type='start_fed',
                            sender=self.ID,
                            receiver=receiver_ID,
                            state=self.state + 1,
                            content=[content]))
            logger.info(
                f'\tServer #{self.ID}: --------start federated training----------- @{self.state // 2}.'
            )
            self.state += 1
        elif self.check_buffer(self.state, self.client_num, check_eval_result) and self.state > 0:
            # Sum up gradient client-wisely and send back
            # FedST:update GNN model on server and
            if not check_eval_result:  # in the training process
                # Perform FedAvg and save the gradient of global embedding to the self.emb_grad_list
                aggregated_num = self._perform_federated_aggregation()

                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    # logger.info(f'Server: Starting evaluation at the end '
                    #             f'of round {self.state - 1}.')
                    self.eval()

                ############################# Update the global GSL module#######################################################
                # update global GSL module
                self.global_GSL.train()
                self.GSL_optimizer.zero_grad()
                glob_emb_grad = torch.cat(self.emb_grad_list, dim=0)  # concatenating the tensor of glob emb gradient
                self.glob_emb.backward(glob_emb_grad)
                self.GSL_optimizer.step()

                # get updated glob node embedding matrix
                self.glob_emb = self.global_GSL(self.node_emb_all)  # get new glob node embedding
                # logger.info(
                #     f'\tServer #{self.ID}: --------Completed update of global GSL module----------- @{self.state // 2}.'
                # )
                ####################################################################################################################

                if self.state < self.total_round_num:
                    # Move to next round of training
                    # logger.info(
                    #     f'----------- Starting a new training round (Round '
                    #     f'#{self.state}) -------------')
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()
                    self.msg_buffer['train'][self.state] = dict()
                    self.staled_msg_buffer.clear()

                    ######################################################################
                    # send the global embedding matrix and updated global model to the corresponding client


                    temp = 0
                    for i in self.client_IDs:
                        i_num = self.client_node_num[i]
                        glob_emb_i = self.glob_emb[temp:temp + i_num].detach()
                        temp += i_num

                        # send glob emd to client i
                        self.comm_manager.send(
                            Message(msg_type='glob_embedding',
                                    sender=self.ID,
                                    receiver=i,
                                    state=self.state,
                                    content=[glob_emb_i]))

                    # send updated model to all client
                    self.broadcast_model_para(msg_type='model_para',
                                              sample_client_num=self.sample_client_num)
                    #############################################################################

                    #################################################
                    # 保存每一轮学到的邻接矩阵，可视化用 #TODO: 待删除
                    # if self.state % 3 ==0:
                    #     temp_adj = self.global_GSL.get_adj()
                    #     torch.save(temp_adj, f'./TEMP/adj_{self.state}.pt')

                    #################################################

                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval()
            else:
                # Receiving enough feedback in the evaluation process
                self._merge_and_format_eval_results()
        else:
            move_on_flag = False

        return move_on_flag

    def _perform_federated_aggregation(self):
        """
        Perform federated aggregation and update the global model
        """
        train_msg_buffer = self.msg_buffer['train'][self.state]
        model = self.models[0]
        aggregator = self.aggregators[0]
        msg_list = list()
        staleness = list()

        ############################################################################
        self.emb_grad_list = list()
        for client_id in self.client_IDs:
            msg_list.append(train_msg_buffer[client_id][:2])
            self.emb_grad_list.append(train_msg_buffer[client_id][2])
        ############################################################################

        # Aggregate
        aggregated_num = len(msg_list)
        agg_info = {
            'client_feedback': msg_list,
            'recover_fun': self.recover_fun,
            'staleness': staleness,
        }
        # logger.info(f'The staleness is {staleness}')
        result = aggregator.aggregate(agg_info)
        # Due to lazy load, we merge two state dict
        merged_param = merge_param_dict(model.state_dict().copy(), result)
        model.load_state_dict(merged_param, strict=False)

        return aggregated_num

    # def _merge_and_format_eval_results(self):
    #     # TODO: 测试用，待删除
    #     """
    #     The behaviors of server when receiving enough evaluating results
    #     """
    #     # Get all the message & aggregate
    #     formatted_eval_res = \
    #         self.merge_eval_results_from_all_clients()
    #     self.history_results = merge_dict(self.history_results,
    #                                       formatted_eval_res)
    #     if self.mode == 'standalone' and \
    #             self._monitor.wandb_online_track and \
    #             self._monitor.use_wandb:
    #         self._monitor.merge_system_metrics_simulation_mode(
    #             file_io=False, from_global_monitors=True)
    #     out_dict = {
    #         'round': [formatted_eval_res['Round']],
    #         'test_acc': [formatted_eval_res['Results_avg']['test_acc']]
    #     }
    #     df = pd.DataFrame(out_dict, columns=out_dict.keys())
    #     df.to_csv(f'./TEMP/result_acc.csv', mode='a', index=False, header=False)
    #     # formatted_eval_res['Results_avg']['test_acc']
    #     self.check_and_save()


class FedGSL_Client(Client):
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cuda:0',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):
        super(FedGSL_Client,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, is_unseen_client, *args, **kwargs)

        # self.data = data.to(device)
        self.data = data['data'].to(device)
        self.device = device
        self.MSELoss = nn.MSELoss()
        self.pretrain_epoch = self._cfg.model.fedgsl.pretrain_epoch
        self.AE = localAE(in_channels=self.data.x.shape[-1], out_channels=self._cfg.model.fedgsl.pretrain_out_channels)

        self.decoder = model

        self.register_handlers('local_pretrain', self.callback_funcs_for_local_pre_train)
        self.register_handlers('glob_embedding', self.callback_funcs_for_glob_emb)
        self.register_handlers('start_fed', self.callback_funcs_for_setup_fedgsl)

        # self.register_handlers('gradient', self.callback_funcs_for_gradient)
        # self.register_handlers('setup', self.callback_funcs_for_setup_fedsage)

    def callback_funcs_for_local_pre_train(self, message: Message):  # 冒号是用来限制message的传入类型
        round, sender, content = message.state, message.sender, message.content

        logger.info(f'\tClient #{self.ID} pre-train start...')
        optimizer_AE = torch.optim.Adam(self.AE.parameters(), self._cfg.model.fedgsl.AE_lr)

        best_loc_loss = 1e9
        best_epoch = 0
        self.AE = self.AE.to(self.device)
        self.loc_label = self.data.y
        if self._cfg.model.fedgsl.update_RawNodeFeature:
            self.loc_emb = self.data.x.to('cuda:0')
            logger.info(f'直接上传原始node features')
        else:
            self.AE.train()
            for i in range(self.pretrain_epoch):
                optimizer_AE.zero_grad()
                loc_emb, loc_reconstruct = self.AE(self.data)
                loss = self.MSELoss(loc_reconstruct, self.data.x)
                if loss <= best_loc_loss:
                    logger.info(
                        f'\tClient #{self.ID} better pre-train_loss:{loss} epoch:{i}.'
                    )
                    self.loc_emb = loc_emb
                    self.final_feat_reconstruct = loc_reconstruct
                    best_loc_loss = loss
                loss.backward()
                optimizer_AE.step()

            with torch.no_grad():
                final_loc_loss = self.MSELoss(self.final_feat_reconstruct, self.data.x)
                logger.info(f'\tClient #{self.ID} final loss:{final_loc_loss}')
                logger.info(f'\tClient #{self.ID} pre-train finish!')
                # # LDP
                # if self._cfg.model.add_noise:
                #     self.feat_emb = add_noise(self.feat_emb, self._cfg.model.LDP_lambda).cuda()
        self.state = round

        # Try clearing GPU cache
        self.AE.cpu()
        torch.cuda.empty_cache()

        # Send the compressed low-dimensional node embedding matrix to the server
        # TODO:确定服务器接收到“compressed_local_node_embedding”消息时，轮数为0（）
        self.comm_manager.send(
            Message(msg_type='compressed_local_node_embedding',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=[self.loc_emb.detach(), self.loc_label.detach()]))
        logger.info(f'\tClient #{self.ID} send local node embedding matrix to the server #{sender}.')

    def callback_funcs_for_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content

        self.trainer.update(content,
                            strict=self._cfg.federate.share_local_model)
        self.state = round
        sample_size, model_para, results = self.trainer.train()
        if self._cfg.federate.share_local_model and not \
                self._cfg.federate.online_aggr:
            model_para = copy.deepcopy(model_para)

        logger.info(
            self._monitor.format_eval_res(results, rnd=self.state,
                                          role='Client #{}'.format(self.ID),
                                          return_raw=True
                                          )
        )

        self.comm_manager.send(
            Message(msg_type='update_GNN_and_globalModel',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, model_para, self.glob_embedding.grad)))
    def callback_funcs_for_glob_emb(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        self.glob_embedding = content[0]
        self.glob_embedding.requires_grad = True
        self.trainer.ctx.glob_emb = self.glob_embedding

    def callback_funcs_for_setup_fedgsl(self, message: Message):
        # 接收服务器第一次传回来的global embedding
        round, sender, content = message.state, message.sender, message.content
        self.glob_emb = content[0].data
        self.glob_emb.requires_grad = True

        self.trainer.ctx.glob_emb = self.glob_emb
        sample_size, model_para, results = self.trainer.train()
        self.state = round
        self.comm_manager.send(
            Message(msg_type='update_GNN_and_globalModel',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, model_para, self.glob_emb.grad)
                    )
        )





# Build Client and Server here for our FedGSL.
def call_my_worker(method):
    if method == 'fedgsl':
        worker_builder = {'client': FedGSL_Client, 'server': FedGSL_Server}
        return worker_builder


register_worker('fedgsl', call_my_worker)
