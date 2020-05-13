from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

from Logger import system_log
from config import system_config

from train import Trainer
from detect.MCDetector import CenterNetDetector

import os 
import time 
import argparse
import cv2 as cv 

parser = argparse.ArgumentParser(description="Demo")
parser.add_argument("Phase", help="train or test", type=str)
parser.add_argument("--test_img", default=None, type=str)
parser.add_argument("--model_path", default=None, type=str)
parser.add_argument("--use_gpu", action='store_true', default=False)
parser.add_argument("--test_folder", default=None, type=str)

args = parser.parse_args()

if __name__ == "__main__":
    phase = args.Phase

    if phase == "train":
        log_file = os.path.join("log/train.log")
        system_log.set_filepath(log_file)

        trainer = Trainer("mbv3_CenterNet", "data/train")
        trainer.run()

    elif phase == "test":
        img_path = args.test_img
        model_path = args.model_path
        device = "gpu" if args.use_gpu else "cpu"

        Detector = CenterNetDetector("mbv3_CenterNet", model_path, device)
        t1 = time.time()
        result = Detector.run(img_path)
        t2 = time.time()



        # write img
        cv_img = cv.imread(img_path)
        for ret in result:
            bbox = ret["bbox"]
            score = ret["score"]
            cv.rectangle(cv_img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[3]), int(bbox[1]+bbox[2])), (0,0,255), thickness=2)
            cv.putText(cv_img, f"score={score:.2f}", (int(bbox[0]+5), int(bbox[1]+12)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), thickness=1)

        cv.imwrite("result.jpg", cv_img)

        print(f"result = {result}, cost time: {(t2-t1):.8f}sec!")
    
    elif phase == "test_batch":
        test_folder = args.test_folder
        model_path = args.model_path
        device = "gpu" if args.use_gpu else "cpu"

        img_list = list(filter(lambda x: ".jpg" in x, os.listdir(test_folder)))
        img_len = len(img_list)

        Detector = CenterNetDetector("mbv3_CenterNet", model_path, device)
        acc = 0
        count = 0

        t1 = time.time()
        for i in range(img_len):
            img_path = os.path.join(test_folder, img_list[i])
            result = Detector.run(img_path)
            print(f"test [{i+1}/{img_len}], result is {result}")
            if not len(result) == 0:
                count += 1
        
        t2 = time.time()
        acc = count / img_len 
        print(f"total is {img_len}, count: {count}, acc: {acc}, cost time: {(t2-t1):.8f}sec!")





    
    