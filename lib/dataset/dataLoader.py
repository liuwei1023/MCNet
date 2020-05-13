from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Logger import system_log
from config import system_config

import os 
import numpy as np 
import torch 
import torch.utils.data as data
import cv2 as cv 
import math
from PIL import Image
import json 
import traceback
import time

from utils.images import draw_msra_gaussian, gaussian_radius
from utils.utils import bbox_xyhw2xyxy

class dataset_loader(data.Dataset):
    def __init__(self, input_folder, transform=None):
        self.imgs_path = []
        self.labels = []
        self.trans = transform

        self.num_class = system_config.Detector['num_class']
        self.max_objs = system_config.Detector['max_ojbs']
        self.down_ration = float(system_config.Detector["down_ration"])

        system_log.WriteLine(f"loading dataset...")

        start = time.time()
        for img_name in filter(lambda x: ".jpg" in x, os.listdir(input_folder)):
            file_name = img_name.split(".")[0]
            label_name = file_name + ".json"
            img_path = os.path.join(input_folder, img_name)
            label_path = os.path.join(input_folder, label_name)

            try:
                with open(label_path, 'r') as f:
                    self.labels.append(json.load(f))

                self.imgs_path.append(img_path)
            
            except:
                system_log.WriteLine(f"load {label_name} error. skip the image.")
                exstr = traceback.format_exc()
                system_log.WriteLine(f"{exstr}")
        end = time.time()

        system_log.WriteLine(f"loading dataset done. total images: {len(self.imgs_path)}. cost time: {(end-start):.8f} sec.")

    def image_padding(self, img):
        h,w,c = img.shape
        if not h%4==0:
            h_border = 4 - (h%4)
            img = cv.copyMakeBorder(img, 0, h_border, 0, 0, cv.BORDER_REPLICATE)
        if not w%4==0:
            w_border = 4 - (w%4)
            img = cv.copyMakeBorder(img, 0, 0, 0, w_border, cv.BORDER_REPLICATE)

        return img 


    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        label = self.labels[index]

        anns = label["anns"]
        num_objs = min(len(anns), self.max_objs)

        cv_img = self.image_padding(cv.imread(img_path))
        input_h, input_w = cv_img.shape[0], cv_img.shape[1]
        output_h, output_w = int(input_h//self.down_ration), int(input_w//self.down_ration)

        if self.trans:
            img = Image.fromarray(cv.cvtColor(cv_img, cv.COLOR_BGR2RGB))
            img = self.trans(img)
            cv_img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)

        inp = (cv_img.astype(np.float32) / 255.)
        inp = inp.transpose(2, 0, 1)


        hm = np.zeros((self.num_class, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        ind = np.zeros((self.max_objs), dtype=np.int64)

        draw_gaussian = draw_msra_gaussian

        for k in range(num_objs):
            ann = anns[k]
            bbox = bbox_xyhw2xyxy(ann["bbox"])      # xyhw to xyxy
            bbox = [i / self.down_ration for i in bbox]
            

            cls_id = int(ann["cls_id"])

            h,w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1

        ret = {'input': inp, 'hm': hm, 'reg': reg, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}

        return ret

    def __len__(self):
        return len(self.imgs_path)


