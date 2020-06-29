import os
import numpy as np
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from loss.triplet import TripletLoss
from loss.center_loss import CenterLoss
from loss.crossentropy_ls import CrossEntropyLoss_ls, FocalLoss


class gradcam_Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(gradcam_Loss, self).__init__()
        print('[INFO] Making gradcam_Loss...')

        self.nGPU = args.nGPU
        self.args = args
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.gradcam_loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'CrossEntropy':
                loss_function = nn.CrossEntropyLoss()
            elif loss_type == 'Triplet':
                loss_function = TripletLoss(args.margin)

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function
            })

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        self.device = torch.device('cuda')
        self.loss_module.to(self.device)

        if args.nGPU > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.nGPU)
            )

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def forward(self, outputs, labels):
        losses = []
        for i, l in enumerate(self.loss):
            if l['type'] == 'CrossEntropy':
                loss = [l['function'](output, labels) for output in outputs[-1]]
                loss = sum(loss) / len(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
            elif l['type'] == 'Triplet':
                # loss = l['function'](outputs[0], labels)
                loss = [l['function'](output, labels) for output in outputs[1:-1]]
                loss = sum(loss) / len(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
        loss_sum = sum(losses)

        return loss_sum

    def get_loss_module(self):
        if self.nGPU == 1:
            return self.loss_module
        else:
            return self.loss_module.module

