from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import system_config

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import _transpose_and_gather_feat


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()
  
    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss

class CenterNetLoss(torch.nn.Module):
    def __init__(self):
        super(CenterNetLoss, self).__init__()

        self.crit = torch.nn.MSELoss()
        self.crit_reg = RegL1Loss()

    def forward(self, output, batch):
        hm_loss, wh_loss, off_loss = 0, 0, 0

        hm_loss += self.crit(output['hm'], batch['hm'])
        if system_config.DLTrain['wh_weight'] > 0:
            wh_loss += self.crit_reg(output['wh'], batch['reg_mask'],batch['ind'], batch['wh'])
        if system_config.DLTrain['off_weight'] > 0:
            off_loss += self.crit_reg(output['reg'], batch['reg_mask'],batch['ind'], batch['reg'])

        loss = system_config.DLTrain['hm_weight'] * hm_loss + system_config.DLTrain['wh_weight'] * wh_loss + system_config.DLTrain['off_weight'] * off_loss

        loss_stats = {"loss": loss, "hm_loss": hm_loss,
                      "wh_loss": wh_loss, "off_loss": off_loss}

        return loss, loss_stats
        
