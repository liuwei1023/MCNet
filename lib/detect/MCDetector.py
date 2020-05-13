from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from Logger import system_log
from config import system_config

import torch 
import time 
import cv2 as cv 
import torch.nn.functional as F

from .base_detector import BaseDetector
from utils.utils import _transpose_and_gather_feat, _gather_feat, bbox_xyxy2xyhw

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


class CenterNetDetector(BaseDetector):
    def __init__(self, model_name, model_path, device='gpu'):
        super(CenterNetDetector, self).__init__(model_name, model_path, device)
        self.threshold = system_config.Detector['threshold']

    def decode(self, heat, wh, reg=None, K=100):
        batch, cat, height, width = heat.size()

        # heat = torch.sigmoid(heat)
        # perform nms on heatmaps
        heat = _nms(heat)
        
        scores, inds, clses, ys, xs = _topk(heat, K=K)
        if reg is not None:
            reg = _transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, K, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5
        wh = _transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)
        clses  = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)
        bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2, 
                            ys + wh[..., 1:2] / 2], dim=2)
        detections = torch.cat([bboxes, scores, clses], dim=2)
        
        return detections

    def process(self, tensor):
        with torch.no_grad():
            output_dict = self.model(tensor)     # output_dict: {"hm": hm, "reg": reg, "wh": wh}, hm.shape: n,cls_num,output_h,output_w,  reg.shape: n,2,output_h,output_w,  wh.shape: n,2,output_h,output_w

            hm = output_dict["hm"]
            reg = output_dict["reg"]
            wh = output_dict["wh"]

            detections = self.decode(hm, wh, reg=reg, K=system_config.Detector['max_ojbs'])     # detections.shape: n,K,6. {x1,y1,x2,y2,score,cls_id}


        return detections

    def merge_output(self, dets):
        # dets.shape: n,K,6. {x1,y1,x2,y2,score,cls_id}
        down_ration = system_config.Detector["down_ration"]
        if self.device == "gpu":
            dets = dets.cpu()

        dets = dets.numpy()[0]      # shape: K,6
        obj_num, _ = dets.shape

        result = []     # result: [{"bbox":[x,y,h,w], "score":score, "cls_id":cls_id}, ...]

        for i in range(obj_num):
            if dets[i][4] > self.threshold:
                det = {}
                bbox = [dets[i][0]*down_ration, dets[i][1]*down_ration, dets[i][2]*down_ration, dets[i][3]*down_ration]
                bbox = bbox_xyxy2xyhw(bbox)
                score = dets[i][4]
                cls_id = dets[i][5]

                det["bbox"] = bbox
                det["score"] = score
                det["cls_id"] = cls_id

                result.append(det)

        return result










