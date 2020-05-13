from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Logger import system_log

import torch


def bbox_xyhw2xyxy(bbox):
    x,y,h,w = bbox
    
    x1 = x 
    y1 = y 
    x2 = x1 + w 
    y2 = y1 + h 

    return [x1,y1,x2,y2]

def bbox_xyxy2xyhw(bbox):
    x1,y1,x2,y2 = bbox

    x = x1 
    y = y1 
    h = y2 - y1 if y2>y1 else 0
    w = x2 - x1 if x2>x1 else 0

    return [x,y,h,w]

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

        if self.sum > 999999:
            self.sum = self.avg
            self.count = 1