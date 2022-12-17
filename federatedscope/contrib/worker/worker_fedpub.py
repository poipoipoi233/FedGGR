import torch
import logging
import copy
import numpy as np
import pickle
import sys

from federatedscope.register import register_worker
from federatedscope.core.message import Message
from federatedscope.core.workers.server import Server
from federatedscope.core.workers.client import Client
from federatedscope.core.auxiliaries.sampler_builder import get_sampler

logger = logging.getLogger(__name__)


class FEDPUB_Server(Server):
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        super(FEDPUB_Server,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)
        data_shape = next(iter(data['train'])).x.shape
        self.proxy_data = self.get_proxy_data(data_shape[1])  # 代理数据：生成的随机图
        self.update_lists = []
        self.sim_matrices = []

        # TODO: 查看如何在这里读取客户端采样里，确认采样率初始为0（）
        n_connected = self.client_num
        self.avg_sim_matrix = np.zeros(shape=(n_connected, n_connected))

    def get_proxy_data(self, n_feat):
        import networkx as nx
        from torch_geometric.utils import from_networkx

        num_graphs = self._cfg.model.fedpub.n_proxy  # 初始为5，很奇怪，似乎该参数时写死的？
        num_nodes = 100
        G = nx.random_partition_graph(
            [num_nodes] * num_graphs, p_in=0.1, p_out=0,
            seed=self._cfg.seed)  # 返回一个随机划分的具有固定大小s的社区图，节点在同一组内相连的概率是p_in,跨组相连的概率是p_out；注意此代码中设置跨组相连的概率是0
        data = from_networkx(G)
        data.x = torch.normal(mean=0, std=1, size=(num_nodes * num_graphs, n_feat))  # 随机生成每个节点的特征，和Cora节点特征数相同
        return data

    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
        # key code
        """
        To check the message_buffer. When enough messages are receiving,
        some events (such as perform aggregation, evaluation, and move to
        the next training round) would be triggered.

        Arguments:
            check_eval_result (bool): If True, check the message buffer for
            evaluation; and check the message buffer for training otherwise.
        """
        # if min_received_num is None:
        #     if self._cfg.asyn.use:
        #         min_received_num = self._cfg.asyn.min_received_num
        #     else:
        #         min_received_num = self._cfg.federate.sample_client_num
        # assert min_received_num <= self.sample_client_num
        min_received_num = len(self.comm_manager.get_neighbors().keys())

        if check_eval_result and self._cfg.federate.mode.lower(
        ) == "standalone":
            # in evaluation stage and standalone simulation mode, we assume
            # strong synchronization that receives responses from all clients
            min_received_num = len(self.comm_manager.get_neighbors().keys())  # TODO: 查看comm_manger初始化，弄懂什么时候增加的邻居(√)：Server类 callback_funcs_for_join_in（）

        move_on_flag = True  # To record whether moving to a new training
        # round or finishing the evaluation
        if self.check_buffer(self.state, min_received_num, check_eval_result):
            if not check_eval_result:
                # Receiving enough feedback in the training process
                client_IDs = [i for i in range(1, min_received_num + 1)]
                self.client_IDs= client_IDs
                aggr_local_model_weights = self._perform_federated_aggregation()

                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    logger.info(f'Server: Starting evaluation at the end '
                                f'of round {self.state - 1}.')
                    # self.eval() #TODO: 此处有误，server会将非个性化全局模型传回client (√)
                    for i in self.client_IDs:
                        self.comm_manager.send(
                            Message(msg_type='evaluate',
                                    sender=self.ID,
                                    receiver=i,
                                    state=min(self.state, self.total_round_num),
                                    timestamp=self.cur_timestamp,
                                    content=aggr_local_model_weights[i]))
                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()
                    self.msg_buffer['train'][self.state] = dict()
                    self.staled_msg_buffer.clear()
                    # Start a new training round
                    for i in self.client_IDs:
                        self.comm_manager.send(
                            Message(msg_type='model_para',
                                    sender=self.ID,
                                    receiver=i,
                                    state=self.state,
                                    content=aggr_local_model_weights[i]))

                    # self._start_new_training_round(aggregated_num)
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

    def trigger_for_start(self):
        #Server类 callback_funcs_for_join_in()的结尾会调用该方法
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
                '-----------Server distributes proxy data (random community graph) to each client (Round #{:d})-----------'.
                format(self.state)
            )
            self.comm_manager.send(
                Message(msg_type='proxy_data',
                        sender=self.ID,
                        receiver=list(self.comm_manager.neighbors.keys()),
                        state=self.state,
                        timestamp=self.cur_timestamp,
                        content=self.proxy_data))
            ####################################################################################


            logger.info(
                '----------- Starting training (Round #{:d}) -------------'.
                format(self.state))
            self.broadcast_model_para(msg_type='model_para',
                                      sample_client_num=self.sample_client_num)

    def _perform_federated_aggregation(self):
        """
        Perform federated aggregation and update the global model
        """
        train_msg_buffer = self.msg_buffer['train'][self.state]
        for model_idx in range(self.model_num):
            aggregator = self.aggregators[model_idx]
            msg_list = list()
            staleness = list()
            for client_id in self.client_IDs:#这里的keys不包括server编号0
                if self.model_num == 1:
                    msg_list.append(train_msg_buffer[client_id])
                else:
                    train_data_size, model_para_multiple = \
                        train_msg_buffer[client_id]
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))
            agg_info = {
                'client_feedback': msg_list,
                'recover_fun': self.recover_fun,
                'staleness': staleness,
            }
            result = aggregator.aggregate(agg_info)
        return result


class FEDPUB_Clinet(Client):
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):
        super(FEDPUB_Clinet, self).__init__(ID, server_id, state, config, data, model, device,
                                            strategy, is_unseen_client, *args, **kwargs)
        # for receiving proxy data
        self.register_handlers(
            msg_type='proxy_data',
            callback_func=self.callback_for_proxy_data)

    def callback_for_proxy_data(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        # self.proxy_data = content
        self.trainer.ctx.proxy_data = content
        logger.info(f'\tClient #{self.ID} has received the proxy data')

    def callback_funcs_for_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        # Cache old W
        W_old = copy.deepcopy(content)
        self.trainer.ctx.W_old = W_old  # TODO: 确保传到trainer里的真的是上一轮的W_old ( )
        self.trainer.ctx.curr_rnd = round

        self.trainer.update(content)

        self.state = round
        sample_size, model_para, results = self.trainer.train()
        if self._cfg.federate.share_local_model and not \
                self._cfg.federate.online_aggr:
            model_para = copy.deepcopy(model_para)
        logger.info(
            self._monitor.format_eval_res(results,
                                          rnd=self.state,
                                          role='Client #{}'.format(self.ID)))
        proxy_output = self.trainer.get_proxy_output()
        # self.comm_manager.send(
        #     Message(msg_type='model_para',
        #             sender=self.ID,
        #             receiver=[sender],
        #             state=self.state,
        #             content=(sample_size, model_para)))

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, model_para, proxy_output)))


# Build Client and Server here for FEDPUB.
def call_my_worker(method):
    if method == 'fedpub':
        worker_builder = {'client': FEDPUB_Clinet, 'server': FEDPUB_Server}
        return worker_builder


register_worker('fedpub', call_my_worker)
