import copy
import logging
import time
import pdb
import numpy as np
import torch
from torch import nn
from fedml_api.model.cv.cnn_meta import Meta_net
import torch.nn.functional as F

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class Soft_Distillation_Loss(nn.Module):
    def __init__(self, lambda_balancing: float, alpha, beta):
        super(Soft_Distillation_Loss, self).__init__()

        self.lambda_balancing = lambda_balancing
        self.alpha = alpha
        self.beta = beta
        self.CE_student = nn.CrossEntropyLoss()

    def forward(self, teacher_y, student_y, y, temperature, epoch):
        # # DKD
        loss_ce = self.lambda_balancing * F.cross_entropy(student_y, y.long())
        loss_dkd = self.lambda_balancing * dkd_loss(student_y, teacher_y, y, self.alpha, self.beta, temperature)

        return loss_ce + loss_dkd


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target.long())
    other_mask = _get_other_mask(logits_student, target.long())
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, size_average=False)
            * (temperature ** 2)
            / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
            * (temperature ** 2)
            / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output) / 2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None, logger=None):
        super().__init__(model, args)
        self.args = args
        self.logger = logger

    def set_masks(self, masks):
        self.masks = masks

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_trainable_params(self):
        dict = {}
        for name, param in self.model.named_parameters():
            dict[name] = param
        return dict


    def train_distill(self, train_data, device, w_global, nei_indexs, teacher_model, args, round):

        student_model = self.model
        student_model.to(device)
        student_model.train()

        criterion = Soft_Distillation_Loss(self.args.lambda_balancing, self.args.alpha, self.args.beta)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=args.lr * (args.lr_decay ** round), momentum=args.momentum, weight_decay=args.wd)
        avg_loss = []
        for epoch in range(args.epochs):
            epoch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                student_model.zero_grad()
                student_pred = student_model.forward(x)
                teacher_model.load_state_dict(w_global[nei_indexs[0]])
                teacher_model.preprocess_flag = False
                for parameter in teacher_model.parameters():
                    parameter.requires_grad = False
                teacher_model.eval()
                teacher_model.cuda()
                with torch.no_grad():
                    teacher_pred = teacher_model.forward(x)

                loss = criterion.forward(teacher_pred, student_pred, labels, self.args.temperature, epoch)
                loss.backward()
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                optimizer.step()
                epoch_loss.append(loss.item())
            avg_loss.append(sum(epoch_loss) / len(epoch_loss))
            self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
        return sum(avg_loss) / len(avg_loss)

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

    def train(self, train_data, device, args, round):
        # torch.manual_seed(0)
        model = self.model
        model.to(device)
        model.train()
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=args.lr * (args.lr_decay ** round), momentum=args.momentum,
                                        weight_decay=args.wd)
        avg_loss = []
        for epoch in range(args.epochs):
            epoch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()

                log_probs = model.forward(x)
                loss = criterion(log_probs, labels.long())
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
