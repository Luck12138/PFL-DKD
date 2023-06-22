import copy
import logging
import math
import time
import pdb
import numpy as np
import torch
from scipy.stats import bernoulli

class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer, logger):
        self.logger = logger
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.logger.info("self.local_sample_number = " + str(self.local_sample_number))
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.epsilon = 0.01

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global, round):

        self.model_trainer.set_model_params(w_global)
        self.model_trainer.set_id(self.client_idx)
        avg_loss = self.model_trainer.train(self.local_training_data, self.device, self.args, round)
        return avg_loss

    def local_test(self, w, b_use_test_dataset=True):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        self.model_trainer.set_model_params(w)
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics

    def upload_mask(self, n_samples):
        """
            Mask samples to be uploaded to the server.
        """
        param_dict = dict()
        num_params = 0.0
        num_ones = 0.0
        with torch.no_grad():
            for _ in range(n_samples):
                for k, v in self.model_trainer.get_model().items():
                    if 'mask' in k:
                        theta = torch.sigmoid(v).cpu().numpy()
                        updates_s = bernoulli.rvs(theta)
                        updates_s = np.where(updates_s == 0, self.epsilon, updates_s)
                        updates_s = np.where(updates_s == 1, 1-self.epsilon, updates_s)

                        # Keep track of the frequency of 1s.
                        num_params += updates_s.size
                        num_ones += np.sum(updates_s)

                        if param_dict.get(k) is None:
                            param_dict[k] = torch.tensor(updates_s, device=self.device)
                        else:
                            param_dict[k] += torch.tensor(updates_s, device=self.device)
                    else:
                        param_dict[k] = v
        local_freq = num_ones / num_params
        return param_dict, local_freq, num_params
