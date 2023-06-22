# -*- coding:utf-8 -*-
# @FileName  :fedistill_api.py
# @Time      :2022/11/6 10:47
# @Author    :lucas
from fedml_api.standalone.fedistill.client import Client
import copy
import numpy as np


class FedistillAPI(object):
    def __init__(self, dataset, device, args, model_trainer, teacher_trainer, logger):
        self.logger = logger
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.model_trainer = model_trainer
        self.teacher_trainer = teacher_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        self.init_stat_info()

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        self.logger.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer, self.logger)
            self.client_list.append(c)
        self.logger.info("############setup_clients (END)#############")

    def train(self):

        w_per_mdls = []
        # 初始化
        for clnt in range(self.args.client_num_in_total):
            w_per_mdls.append(copy.deepcopy(self.model_trainer.get_model_params()))

        # select neighbors
        neighbors = {}
        for clnt in range(self.args.client_num_in_total):
            nei_indexs = self._benefit_choose(clnt, self.args.client_num_in_total,
                                              self.args.client_num_per_round, self.args.cs)
            neighbors[clnt] = nei_indexs

        for round_idx in range(self.args.comm_round):
            self.logger.info("################Communication round : {}".format(round_idx))

            w_per_mdls_lstrd = copy.deepcopy(w_per_mdls)

            training_loss = []
            for clnt_idx in range(self.args.client_num_in_total):
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(round_idx, clnt_idx))
                #
                # #select neighbors
                # nei_indexs = self._benefit_choose(clnt_idx, self.args.client_num_in_total,
                #                                   self.args.client_num_per_round,  self.args.cs, None)
                # # 如果不是全选，则补上当前clint，进行聚合操作
                # if self.args.client_num_in_total != self.args.client_num_per_round:
                #     nei_indexs = np.append(nei_indexs, clnt_idx)
                nei_indexs = neighbors[clnt_idx]
                nei_indexs = np.sort(nei_indexs)
                # update dataset
                client = self.client_list[clnt_idx]
                if clnt_idx == 0:
                    w_per, training_flops, num_comm_params, avg_loss = client.train(w_per_mdls_lstrd[0],
                                                                                    round_idx)
                else:
                    w_per, training_flops, num_comm_params, avg_loss = client.train_distill(w_per_mdls_lstrd,
                                                                                            nei_indexs,
                                                                                            self.teacher_trainer,
                                                                                            round_idx)
                w_per_mdls[clnt_idx] = copy.deepcopy(w_per)
                # self.logger.info("local weights = " + str(w))
                training_loss.append(avg_loss)
                self.stat_info["sum_training_flops"] += training_flops
                self.stat_info["sum_comm_params"] += num_comm_params

                cln_train_loss = {str(clnt_idx) + '_training_loss': avg_loss}
                self.logger.info(cln_train_loss)

            self.logger.info("################Average Training Loss : {}".format(round_idx))
            train_loss = {'training_loss': sum(training_loss) / len(training_loss)}
            self.logger.info(train_loss)
            # self.logger.info('CM_AVG({}) \tTraining_Loss: {:.6f}'.format(
            #     round, sum(training_loss) / len(training_loss)))
            self._test_on_all_clients(w_per_mdls, round_idx)
            # self._test_on_all_clients_avg(w_per_mdls, round_idx)


    def _benefit_choose(self, cur_clnt, client_num_in_total, client_num_per_round, cs):
        if client_num_in_total == client_num_per_round:
            # If one can communicate with all others and there is no bandwidth limit
            client_indexes = [client_index for client_index in range(client_num_in_total)]
            return client_indexes

        if cs == "random":
            # Random selection of available clients
            num_clients = min(client_num_per_round, client_num_in_total)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
            while cur_clnt in client_indexes:
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)

        elif cs == "ring":
            # Ring Topology in Decentralized setting
            left = (cur_clnt - 1 + client_num_in_total) % client_num_in_total
            right = (cur_clnt + 1) % client_num_in_total
            client_indexes = np.asarray([left, right])

        elif cs == "full":
            active_ths_rnd = np.random.choice([0, 1], size=client_num_in_total,
                                              p=[0, 1.0])
            # Fully-connected Topology in Decentralized setting
            client_indexes = np.array(np.where(active_ths_rnd == 1)).squeeze()
            client_indexes = np.delete(client_indexes, int(np.where(client_indexes == cur_clnt)[0]))
        return client_indexes

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, _) = w_locals[idx]
            training_num += sample_num
        w_global = {}
        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    w_global[k] = local_model_params[k] * w
                else:
                    w_global[k] += local_model_params[k] * w
        return w_global

    def _test_on_all_clients(self, w_per_mdls, round_idx):

        self.logger.info("################global_test_on_all_clients : {}".format(round_idx))

        # p_test_metrics = {
        #     'num_samples': [],
        #     'num_correct': [],
        #     'losses': []
        # }

        for client_idx in range(self.args.client_num_in_total):
            # test data
            client = self.client_list[client_idx]

            p_test_local_metrics = client.local_test(w_per_mdls[client_idx], True)
            test_num = p_test_local_metrics['test_total']
            test_correct = p_test_local_metrics['test_correct']
            test_loss = p_test_local_metrics['test_loss']
            # p_test_metrics['num_samples'].append(copy.deepcopy(p_test_local_metrics['test_total']))
            # p_test_metrics['num_correct'].append(copy.deepcopy(p_test_local_metrics['test_correct']))
            # p_test_metrics['losses'].append(copy.deepcopy(p_test_local_metrics['test_loss']))

            p_test_local_acc = test_correct / test_num
            p_test_local_loss = test_loss / test_num

            # p_test_local_acc = np.array(p_test_metrics['num_correct']) / np.array(p_test_metrics['num_samples'])
            # p_test_local_loss=np.array(p_test_metrics['losses'])/np.array(p_test_metrics['num_samples'])
            per_test_loss = {str(client_idx) + '_test_loss': p_test_local_loss}
            per_test_acc = {str(client_idx) + '_test_acc': round(p_test_local_acc, 4)}
            self.logger.info(per_test_loss)
            self.logger.info(per_test_acc)

        # test on test dataset

        # p_test_acc = sum(
        #     [np.array(p_test_metrics['num_correct'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
        #      range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        # p_test_loss = sum([np.array(p_test_metrics['losses'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in range(self.args.client_num_in_total)]) / self.args.client_num_in_total

        # stats = {'person_test_acc': p_test_acc, 'person_test_loss': p_test_loss}
        # self.stat_info["person_test_acc"].append(p_test_acc)

    def _test_on_all_clients_avg(self, w_per_mdls, round_idx):

        self.logger.info("################global_test_on_all_clients : {}".format(round_idx))

        # test client 0
        client = self.client_list[0]

        p_test_local_metrics_0 = client.local_test(w_per_mdls[0], True)
        test_num_0 = p_test_local_metrics_0['test_total']
        test_correct_0 = p_test_local_metrics_0['test_correct']
        test_loss_0 = p_test_local_metrics_0['test_loss']
        p_test_local_acc_0 = test_correct_0 / test_num_0
        p_test_local_loss_0 = test_loss_0 / test_num_0
        per_test_loss_0 = {str(0) + '_test_loss': p_test_local_loss_0}
        per_test_acc_0 = {str(0) + '_test_acc': round(p_test_local_acc_0, 4)}
        self.logger.info(per_test_loss_0)
        self.logger.info(per_test_acc_0)

        # test client 1-C
        p_test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }
        for client_idx in range(1,self.args.client_num_in_total):
            # test data
            client = self.client_list[client_idx]
            p_test_local_metrics = client.local_test(w_per_mdls[client_idx], True)
            p_test_metrics['num_samples'].append(copy.deepcopy(p_test_local_metrics['test_total']))
            p_test_metrics['num_correct'].append(copy.deepcopy(p_test_local_metrics['test_correct']))
            p_test_metrics['losses'].append(copy.deepcopy(p_test_local_metrics['test_loss']))

        p_test_acc = sum(
            [np.array(p_test_metrics['num_correct'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
             range(self.args.client_num_in_total-1)]) / (self.args.client_num_in_total-1)
        p_test_loss = sum(
            [np.array(p_test_metrics['losses'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
             range(self.args.client_num_in_total-1)]) / (self.args.client_num_in_total-1)

        stats = {'person_test_acc': round(p_test_acc, 4), 'person_test_loss': round(p_test_loss, 4)}
        # self.stat_info["person_test_acc"].append(p_test_acc)
        self.logger.info(stats)


    def init_stat_info(self):
        self.stat_info = {}
        self.stat_info["sum_comm_params"] = 0
        self.stat_info["sum_training_flops"] = 0
        self.stat_info["avg_inference_flops"] = 0
        self.stat_info["global_test_acc"] = []
        self.stat_info["person_test_acc"] = []
        self.stat_info["final_masks"] = []
