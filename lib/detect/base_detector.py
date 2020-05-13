from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Logger import system_log

import cv2 as cv 
import numpy as np 
import torch 
import time 

from models.model import create_model, load_model

class BaseDetector(object):
    def __init__(self, model_name, model_path, device='gpu'):
        self.model = create_model(model_name)
        self.model = load_model(self.model, model_path, device=device)
        self.model.eval()
        self.device = device


    def pre_process(self, image):
        image = image.astype(np.float32).transpose(2,0,1)
        image_tensor = torch.Tensor(image/255.)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        if self.device == "gpu":
            image_tensor = image_tensor.cuda()
        
        return image_tensor


    def process(self, tensor):
        raise NotImplementedError

    def merge_output(self, dets):
        raise NotImplementedError

    def run(self, image_path):
        load_time, pre_time, forward_time, merge_time, total_time = 0, 0, 0, 0, 0

        start_time = time.time()
        cv_img = cv.imread(image_path)
        loaded_time = time.time()
        load_time = loaded_time - start_time

        result = []
        
        if cv_img is not None:
            input_tensor = self.pre_process(cv_img)
            preprocess_time = time.time()
            pre_time = preprocess_time - loaded_time

            output_tensor = self.process(input_tensor).detach()
            net_forward_time = time.time()
            forward_time = net_forward_time-preprocess_time

            result = self.merge_output(output_tensor)       # result: [{"bbox":[x,y,h,w], "score":score, "cls_id":cls_id}, ...]
            merged_time = time.time()
            merge_time = merged_time - net_forward_time
            total_time = merged_time - start_time


        else:
            system_log.WriteLine(f"ERROR: read image {image_path} fail. please check if the path is correct.")


        system_log.WriteLine(f"process image {image_path} done, load_time: {load_time:.8f}sec, "
                              "preprocess_time: {pre_time:.8f}sec, net_forward_time: {forward_time:.8f}sec, "
                              "merge_output_time: {merge_time:.8f}sec,  total time: {total_time:.8f}sec")
        return result

