# -*- coding:utf-8 -*-
# @FileName  :main_pfldkd.py
# @Time      :2022/11/5 15:14
# @Author    :lucas
import argparse
import logging
import os
import random
import sys
import pdb
import numpy as np
import torch
import copy

from fedml_api.data_preprocessing.cifar10.load_data import load_partition_cifar10
from fedml_api.data_preprocessing.cifar100.load_data import load_partition_cifar100
from fedml_api.data_preprocessing.mnist.load_data import load_partition_mnist
from fedml_api.model.cv.cnn_mnist import CNNMnist, MLP
from fedml_api.model.cv.lenet5 import LeNet5
from fedml_api.standalone.dfldkd.pfldkd_api import FedistillAPI
from fedml_api.standalone.dfldkd.my_model_trainer import MyModelTrainer

sys.path.insert(0, os.path.abspath("/Date/FL/PFL-DKD/"))
# sys.path.insert(0, os.path.abspath("/home/featurize/"))
from fedml_api.model.cv.vgg import vgg11
from fedml_api.model.cv.resnet import customized_resnet18, tiny_resnet18
from fedml_api.model.cv.cnn_cifar10 import cnn_cifar10, cnn_cifar100


def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, mode='w', encoding='UTF-8')
    handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='resnet18', metavar='N',
                        help="network architecture, supporting 'cnn_cifar10', 'cnn_cifar100', 'resnet18', 'vgg11'")

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--momentum', type=float, default=0, metavar='N',
                        help='momentum')

    parser.add_argument('--data_dir', type=str, default='/Date/FL/PFL-DKD/data/',
                        help='data directory, please feel free to change the directory to the right place')

    parser.add_argument('--partition_method', type=str, default='dir', metavar='N',
                        help="current supporting three types of data partition, one called 'dir' short for Dirichlet"
                             "one called 'n_cls' short for how many classes allocated for each client"
                             "and one called 'my_part' for partitioning all clients into PA shards with default latent Dir=0.3 distribution")

    parser.add_argument('--partition_alpha', type=float, default=0.3, metavar='PA',
                        help='available parameters for data partition method')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='local batch size for training')

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')

    parser.add_argument('--lr_decay', type=float, default=0.998, metavar='LR_decay',
                        help='learning rate decay (default: 0.998)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=5e-4)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='local training epochs for each client')

    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--frac', type=float, default=0.1, metavar='NN',
                        help='selection fraction each round')

    parser.add_argument('--comm_round', type=int, default=100,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    parser.add_argument("--tag", type=str, default="test")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cs", type=str, default='full')
    parser.add_argument("--active", type=float, default=1.0)

    # kd
    parser.add_argument('--lambda_balancing', type=float, default=1,
                        help='lambda_balancing')
    parser.add_argument('--temperature', type=float, default=7,
                        help='temperature')
    parser.add_argument('--alpha', type=float, default=1,
                        help='alpha')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta')
    return parser


def load_data(args, dataset_name):
    if dataset_name == "cifar10":
        args.data_dir += "cifar10"
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_cifar10(args.data_dir, args.partition_method,
                                           args.partition_alpha, args.client_num_in_total, args.batch_size, logger)
    else:
        if dataset_name == "cifar100":
            args.data_dir += "cifar100"
            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = load_partition_cifar100(args.data_dir, args.partition_method,
                                                args.partition_alpha, args.client_num_in_total,
                                                args.batch_size, logger)
        elif dataset_name == "mnist":
            args.data_dir += "mnist"

            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = load_partition_mnist(args.data_dir, args.partition_method,
                                             args.partition_alpha, args.client_num_in_total,
                                             args.batch_size, logger)
    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def create_model(args, model_name, class_num, logger):
    logger.info("create_model. model_name = %s" % (model_name))
    model = None
    if model_name == "lenet5":
        model = LeNet5(class_num)
    elif model_name == "cnn_cifar10":
        model = cnn_cifar10()
    elif model_name == "cnn_cifar100":
        model = cnn_cifar100()
    elif model_name == "resnet18" and args.dataset != 'tiny':
        model = customized_resnet18(class_num=class_num)
    elif model_name == "resnet18" and args.dataset == 'tiny':
        model = tiny_resnet18(class_num=class_num)
    elif model_name == "vgg11":
        model = vgg11(class_num)
    elif model_name == "cnn_mnist":
        model = CNNMnist(args)
    elif model_name == "mlp_mnist":
        model = MLP(28 * 28, 64, 10)
    return model


def custom_model_trainer(args, model, logger):
    return MyModelTrainer(model, args, logger)


if __name__ == "__main__":

    parser = add_args(argparse.ArgumentParser(description='Fedistill-standalone'))
    args = parser.parse_args()

    data_partition = args.partition_method
    if data_partition != "homo":
        data_partition += str(args.partition_alpha)
    args.identity = "pfldkd" + "-" + data_partition
    args.client_num_per_round = int(args.client_num_in_total * args.frac)
    args.identity += "-mdl" + args.model
    args.identity += "-cm" + str(args.comm_round) + "-total_clnt" + str(args.client_num_in_total)
    args.identity += "-neighbor" + str(args.client_num_per_round)
    args.identity += '-lambda_balancing' + str(args.lambda_balancing)
    args.identity += '-alpha' + str(args.alpha)
    args.identity += '-beta' + str(args.beta)
    args.identity += '-seed' + str(args.seed)

    cur_dir = os.path.abspath(__file__).rsplit("/", 1)[0]
    log_path = os.path.join(cur_dir, 'LOG/' + args.dataset + '/' + args.identity + '.log')
    logger = logger_config(log_path='LOG/' + args.dataset + '/' + args.identity + '.log', logging_name=args.identity)

    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu))
    logger.info(device)
    logger.info("running at device{}".format(device))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # load data
    dataset = load_data(args, args.dataset)

    # create model.
    model = create_model(args, model_name=args.model, class_num=len(dataset[-1][0]), logger=logger)
    # print(model)
    model_trainer = custom_model_trainer(args, model, logger)
    teacher_trainer = copy.deepcopy(model)
    logger.info(model)
    fedistillAPI = FedistillAPI(dataset, device, args, model_trainer, teacher_trainer, logger)

    fedistillAPI.train()
