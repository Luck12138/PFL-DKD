# -*- coding:utf-8 -*-
# @FileName  :fedpm_api.py
# @Time      :2023/5/10 10:56
# @Author    :lucas
import copy
import logging
import math
import pickle
import random
import pdb
import numpy as np
import torch

from fedml_api.standalone.fedpm.client import Client


class FedPMAPI(object):
    def __init__(self, dataset, device, args, model_trainer, logger):
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
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        self.init_stat_info()
        self.alphas = dict()
        self.betas = dict()
        self.lambda_init = 1
        for k, val in self.model_trainer.model.named_parameters():
            self.alphas[k] = torch.ones_like(val) * self.lambda_init
            self.betas[k] = torch.ones_like(val) * self.lambda_init

    def find_bitrate(self, probs, num_params):
        local_bitrate = 0
        for p in probs:
            local_bitrate += p * math.log2(1 / p)
        return local_bitrate * num_params

    def reset_prior(self):
        """
            Reset to uniform prior, depending on lambda_init.
        """
        self.alphas = dict()
        self.betas = dict()
        for k, val in self.model_trainer.model.named_parameters():
            self.alphas[k] = torch.ones_like(val) * self.lambda_init
            self.betas[k] = torch.ones_like(val) * self.lambda_init

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        self.logger.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer, self.logger)
            self.client_list.append(c)
        self.logger.info("############setup_clients (END)#############")

    def train(self):
        w_global = self.model_trainer.get_model_params()
        w_per_mdls = []
        # 初始化
        # for clnt in range(self.args.client_num_in_total):
        #     w_per_mdls.append(copy.deepcopy(w_global))
        # device = {device} cuda:0apply mask to init weights
        for round_idx in range(self.args.comm_round):
            self.logger.info("################Communication round : {}".format(round_idx))
            w_locals = []
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
            client_indexes = np.sort(client_indexes)

            self.logger.info("client_indexes = " + str(client_indexes))
            training_loss = []
            for cur_clnt in client_indexes:
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(round_idx, cur_clnt))
                # update dataset
                client = self.client_list[cur_clnt]
                # update meta components in personal network
                avg_loss = client.train(copy.deepcopy(w_global), round_idx)
                # w_per_mdls[cur_clnt] = copy.deepcopy(w_per)
                # self.logger.info("local weights = " + str(w))
                training_loss.append(avg_loss)
                self.stat_info["sum_training_flops"] += training_flops
                self.stat_info["sum_comm_params"] += num_comm_params
            # update global meta weights
            p_round, bitrate_round, freq_round = self._aggregate(client_indexes)

            train_loss = {'training_loss': sum(training_loss) / len(training_loss)}
            self.logger.info(train_loss)
            self.logger.info(bitrate_round)
            self.logger.info(freq_round)

            self._test_on_all_clients(self.model_trainer.get_model_params(), round_idx)



    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        self.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _aggregate(self,client_indexes):

        aggregated_weights = self.model_trainer.get_model_params()
        aggregated_p = dict()

        for k, v in self.model_trainer.model.named_parameters():
            if 'mask' in k:
                aggregated_p[k] = torch.zeros_like(v)

        with torch.no_grad():
            self.reset_prior()
            n_samples = 1
            p_update = []
            avg_bitrate = 0
            avg_freq = 0
            for client in client_indexes:
                sampled_mask, local_freq, num_params = self.client_list[client].upload_mask(
                    n_samples=n_samples)
                avg_freq += local_freq
                local_bitrate = self.find_bitrate([local_freq + 1e-50, 1 - local_freq + 1e-50], num_params) + math.log2(num_params)
                avg_bitrate += local_bitrate / num_params
                for k, v in sampled_mask.items():
                    if 'mask' in k:
                        self.alphas[k] += v.cpu()
                        self.betas[k] += (n_samples-v.cpu())
                        # Add layerwise estimated ps for each client
                        p_update.extend(v.cpu().numpy().flatten()/n_samples)

            avg_bitrate = avg_bitrate / len(client_indexes)
            avg_freq = avg_freq / len(client_indexes)

            for k, val in aggregated_weights.items():
                if 'mask' in k:
                    avg_p = (self.alphas[k] - 1) / (self.alphas[k] + self.betas[k] - 2)
                    aggregated_weights[k] = torch.tensor(
                        torch.log(avg_p / (1 - avg_p)),
                        requires_grad=True,
                        device=self.device)

        self.model_trainer.set_model_params(aggregated_weights)
        return np.mean(p_update), avg_bitrate, avg_freq


    def _test_on_all_clients(self, w_global, round_idx):

        self.logger.info("################global_test_on_all_clients : {}".format(round_idx))

        g_test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        p_test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        for client_idx in range(self.args.client_num_in_total):
            # test data
            client = self.client_list[client_idx]
            g_test_local_metrics = client.local_test(w_global, True)
            g_test_metrics['num_samples'].append(copy.deepcopy(g_test_local_metrics['test_total']))
            g_test_metrics['num_correct'].append(copy.deepcopy(g_test_local_metrics['test_correct']))
            g_test_metrics['losses'].append(copy.deepcopy(g_test_local_metrics['test_loss']))

            # p_test_local_metrics = client.local_test(w_per_mdls[client_idx], True)
            # p_test_metrics['num_samples'].append(copy.deepcopy(p_test_local_metrics['test_total']))
            # p_test_metrics['num_correct'].append(copy.deepcopy(p_test_local_metrics['test_correct']))
            # p_test_metrics['losses'].append(copy.deepcopy(p_test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
        # test on test dataset
        g_test_acc = sum(
            [np.array(g_test_metrics['num_correct'][i]) / np.array(g_test_metrics['num_samples'][i]) for i in
             range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        g_test_loss = sum([np.array(g_test_metrics['losses'][i]) / np.array(g_test_metrics['num_samples'][i]) for i in
                           range(self.args.client_num_in_total)]) / self.args.client_num_in_total

        # p_test_acc = sum(
        #     [np.array(p_test_metrics['num_correct'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
        #      range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        # p_test_loss = sum([np.array(p_test_metrics['losses'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
        #                    range(self.args.client_num_in_total)]) / self.args.client_num_in_total

        stats = {'global_test_acc': g_test_acc, 'global_test_loss': g_test_loss}
        self.stat_info["global_test_acc"].append(g_test_acc)
        self.logger.info(stats)

        # stats = {'person_test_acc': p_test_acc, 'person_test_loss': p_test_loss}
        # self.stat_info["person_test_acc"].append(p_test_acc)
        # self.logger.info(stats)

    def record_avg_inference_flops(self, w_global, mask_pers=None):
        inference_flops = []
        for client_idx in range(self.args.client_num_in_total):

            if mask_pers == None:
                inference_flops += [self.model_trainer.count_inference_flops(w_global)]
            else:
                w_per = {}
                for name in mask_pers[client_idx]:
                    w_per[name] = w_global[name] * mask_pers[client_idx][name]
                inference_flops += [self.model_trainer.count_inference_flops(w_per)]
        avg_inference_flops = sum(inference_flops) / len(inference_flops)
        self.stat_info["avg_inference_flops"] = avg_inference_flops

    def init_stat_info(self):
        self.stat_info = {}
        self.stat_info["sum_comm_params"] = 0
        self.stat_info["sum_training_flops"] = 0
        self.stat_info["avg_inference_flops"] = 0
        self.stat_info["global_test_acc"] = []
        self.stat_info["person_test_acc"] = []
        self.stat_info["final_masks"] = []
