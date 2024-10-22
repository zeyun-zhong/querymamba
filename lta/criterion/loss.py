import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import pandas as pd
import numpy as np
import os.path as osp


class MultipCrossEntropyLoss(nn.Module):

    def __init__(self, reduction='mean', ignore_index=-100):
        super(MultipCrossEntropyLoss, self).__init__()

        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        input = input.reshape(-1, input.size(-1))
        target = target.reshape(-1, target.size(-1))

        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

        if self.ignore_index >= 0:
            notice_index = [
                i for i in range(target.shape[-1]) if i != self.ignore_index]
            output = torch.sum(
                -target[:, notice_index] * logsoftmax(input[:, notice_index]),
                dim=1
            )

            if self.reduction == 'mean':
                return torch.mean(output[target[:, self.ignore_index] != 1])
            elif self.reduction == 'sum':
                return torch.sum(output[target[:, self.ignore_index] != 1])
            else:
                return output[target[:, self.ignore_index] != 1]
        else:
            output = torch.sum(-target * logsoftmax(input), dim=1)

            if self.reduction == 'mean':
                return torch.mean(output)
            elif self.reduction == 'sum':
                return torch.sum(output)
            else:
                return output


class MultipCrossEntropyEqualizedLoss(nn.Module):

    def __init__(self, gamma=0.95, lambda_=1.76e-3, reduction='mean',
                 ignore_index=-100,
                 anno_path='annotations/ek100_rulstm/',
                 freq_info=None):
        super(MultipCrossEntropyEqualizedLoss, self).__init__()

        if freq_info is None:
            # get label distribution
            segment_list = pd.read_csv(
                osp.join(anno_path, 'training.csv'),
                names=['id', 'video', 'start_f', 'end_f', 'verb', 'noun', 'action'],
                skipinitialspace=True)
            freq_info = np.zeros((max(segment_list['action']) + 1,))
            assert ignore_index == 0
            for segment in segment_list.iterrows():
                freq_info[segment[1]['action']] += 1.
        freq_info = freq_info / freq_info.sum()
        self.freq_info = torch.FloatTensor(freq_info)

        self.gamma = gamma
        self.lambda_ = lambda_
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        input = input.reshape(-1, input.size(-1))
        target = target.reshape(-1, target.size(-1))

        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

        bg_target = target[:, self.ignore_index]
        notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
        input = input[:, notice_index]
        target = target[:, notice_index]

        weight = input.new_zeros(len(notice_index))
        weight[self.freq_info < self.lambda_] = 1.
        weight = weight.view(1, -1).repeat(input.shape[0], 1)

        eql_w = 1 - (torch.rand_like(target) < self.gamma) * weight * (1 - target)
        input = torch.log(eql_w + 1e-8) + input

        output = torch.sum(-target * logsoftmax(input), dim=1)
        if (bg_target != 1).sum().item() == 0:
            return torch.mean(torch.zeros_like(output))
        if self.reduction == 'mean':
            return torch.mean(output[bg_target != 1])
        elif self.reduction == 'sum':
            return torch.sum(output[bg_target != 1])
        else:
            return output[bg_target != 1]
