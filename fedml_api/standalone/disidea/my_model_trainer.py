import copy
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from fedml_core.trainer.model_trainer import ModelTrainer


class Hard_Distillation_Loss(nn.Module):
    def __init__(self):
        super(Hard_Distillation_Loss, self).__init__()

        self.CE_teacher = nn.CrossEntropyLoss()
        self.CE_student = nn.CrossEntropyLoss()

    def forward(self, teacher_y, student_y, y):
        loss = (1 / 2) * (self.CE_student(student_y, y.long())) + (1 / 2) * (self.CE_teacher(teacher_y, y.long()))

        return loss


class Soft_Distillation_Loss(nn.Module):
    def __init__(self, lambda_balancing: float):
        super(Soft_Distillation_Loss, self).__init__()

        self.lambda_balancing = lambda_balancing

        self.CE_student = nn.CrossEntropyLoss()
        # self.KLD_teacher = nn.KLDivLoss()

    def forward(self, teachers_y, student_y, y, temperature):
        soft_loss = nn.KLDivLoss(reduction="batchmean")  # 不包含softmax操作(所以可以自己设定温度系数)

        alpha = self.lambda_balancing / len(teachers_y)
        ditillation_loss_res = 0.0
        for teach_y in teachers_y:
            ditillation_loss = (soft_loss(F.log_softmax(student_y / temperature, dim=1),
                                          F.softmax(teach_y / temperature, dim=1))) * alpha
            ditillation_loss_res += ditillation_loss

        loss = ((1 - self.lambda_balancing) * self.CE_student(student_y, y.long())) + ditillation_loss_res

        # loss = ((1-self.lambda_balancing) * self.CE_student(student_y, y)) + \
        #        (self.lambda_balancing * (temperature**2) *
        #         self.KLD_teacher(student_y / temperature, teacher_y / temperature))
        # loss = ((1-self.lambda_balancing) * self.CE_student(student_y, y.long())) + \
        #        F.kl_div(F.log_softmax(student_y / temperature, dim=1), F.softmax(teacher_y / temperature, dim=1),
        #                 reduction='batchmean') * (temperature ** 2) * self.lambda_balancing

        return loss
    # def forward(self, teacher_y, student_y, y, temperature):
    #
    #     loss = ((1-self.lambda_balancing) * self.CE_student(student_y, y.long())) + \
    #            F.kl_div(F.log_softmax(student_y / temperature, dim=1), F.softmax(teacher_y / temperature, dim=1),
    #                     reduction='batchmean') * (temperature ** 2) * self.lambda_balancing
    #
    #     return loss



def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


class MyModelTrainer(ModelTrainer):

    def __init__(self, model, args=None, logger=None):
        super().__init__(model, args)
        self.args = args
        self.logger = logger

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_trainable_params(self):
        dict = {}
        for name, param in self.model.named_parameters():
            dict[name] = param
        return dict

    def train(self, train_data, device, args, round):
        model = self.model
        model.to(device)
        model.train()
        criterion = nn.CrossEntropyLoss().to(device)

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=args.lr * (args.lr_decay ** round), momentum=args.momentum, weight_decay=args.wd)
        avg_loss = []
        for epoch in range(args.epochs):
            epoch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model.forward(x)
                loss = criterion(log_probs, labels.long())
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                optimizer.step()
                # self.logger.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))
                epoch_loss.append(loss.item())
            avg_loss.append(sum(epoch_loss) / len(epoch_loss))
            self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
        return sum(avg_loss) / len(avg_loss)

    def train_distill(self, train_data, device, w_global, nei_indexs, teacher_model, args, round):
        # temperatures
        # torch.manual_seed(0)
        student_model = self.model
        student_model.to(device)
        student_model.train()
        # train and update
        # criterion = Hard_Distillation_Loss()
        criterion = Soft_Distillation_Loss(self.args.lambda_balancing)
        # criterion = nn.CrossEntropyLoss().to(device)
        # if args.client_optimizer == "sgd":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=args.lr * (args.lr_decay ** round), momentum=args.momentum, weight_decay=args.wd)
        avg_loss = []
        for epoch in range(args.epochs):
            epoch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                student_model.zero_grad()
                student_pred = student_model.forward(x)
                teacher_preds = []
                for i in nei_indexs:
                    teacher_model.load_state_dict(w_global[i])
                    teacher_model.preprocess_flag = False
                    for parameter in teacher_model.parameters():
                        parameter.requires_grad = False
                    teacher_model.eval()
                    teacher_model.cuda()
                    with torch.no_grad():
                        teacher_pred = teacher_model.forward(x)
                        teacher_preds.append(teacher_pred)

                loss = criterion.forward(teacher_preds, student_pred, labels, self.args.temperature)
                loss.backward()
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                optimizer.step()
                # self.logger.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))
                epoch_loss.append(loss.item())
            avg_loss.append(sum(epoch_loss) / len(epoch_loss))
            self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
        return sum(avg_loss) / len(avg_loss)
    # def train_distill(self, train_data, device, w_global, nei_indexs, teacher_model, args, round):
    #     # temperatures
    #     # torch.manual_seed(0)
    #     student_model = self.model
    #     student_model.to(device)
    #     student_model.train()
    #     teacher_model.load_state_dict(w_global[nei_indexs[0]])
    #     teacher_model.preprocess_flag = False
    #     for parameter in teacher_model.parameters():
    #         parameter.requires_grad = False
    #     teacher_model.eval()
    #     teacher_model.cuda()
    #     # train and update
    #     # criterion = Hard_Distillation_Loss()
    #     criterion = Soft_Distillation_Loss(self.args.lambda_balancing)
    #     # criterion = nn.CrossEntropyLoss().to(device)
    #     # if args.client_optimizer == "sgd":
    #     optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
    #                                 lr=args.lr * (args.lr_decay ** round), momentum=args.momentum, weight_decay=args.wd)
    #     avg_loss = []
    #     for epoch in range(args.epochs):
    #         epoch_loss = []
    #         for batch_idx, (x, labels) in enumerate(train_data):
    #             x, labels = x.to(device), labels.to(device)
    #             student_model.zero_grad()
    #             student_pred = student_model.forward(x)
    #
    #             teacher_pred = None
    #             with torch.no_grad():
    #                 teacher_pred = teacher_model.forward(x)
    #             loss = criterion.forward(teacher_pred, student_pred, labels, self.args.temperature)
    #             loss.backward()
    #             # to avoid nan loss
    #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
    #             optimizer.step()
    #             # self.logger.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #             #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
    #             #            100. * (batch_idx + 1) / len(train_data), loss.item()))
    #             epoch_loss.append(loss.item())
    #         avg_loss.append(sum(epoch_loss) / len(epoch_loss))
    #         self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
    #             self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
    #     return sum(avg_loss) / len(avg_loss)

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target.long())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
