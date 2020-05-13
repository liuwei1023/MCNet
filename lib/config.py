import numpy as np 
import os 
import json 

current_path = os.path.dirname(__file__)

class Config(object):
    def __init__(self):
        self._configs = {}

        # _Detector
        self._configs["_Detector"] = {}
        self._Detector = self._configs["_Detector"]

        self._Detector["max_objs"] = 256
        self._Detector["num_class"] = 1
        self._Detector["threshold"] = 0.8
        self._Detector["down_ration"] = 4

        # _DLTrain
        self._configs["_DLTrain"] = {}
        self._DLTrain = self._configs["_DLTrain"]

        self._DLTrain["train_with_multi_gpu"] = True
        self._DLTrain["hm_weight"] = 1.0
        self._DLTrain["wh_weight"] = 0.1
        self._DLTrain["off_weight"] = 1.0
        self._DLTrain["lr"] = 0.001
        self._DLTrain["batch_size"] = 12
        self._DLTrain["max_epoch"] = 100
        

    @property
    def Detector(self):
        return self._Detector

    @property
    def DLTrain(self):
        return self._DLTrain 

    def update_config(self, new):
        for key in new:
            if key == "Detector":
                for sub_key in new["Detector"]:
                    self._Detector[sub_key] = new["Detector"][sub_key]
            elif key == "DLTrain":
                for sub_key in new["DLTrain"]:
                    self._DLTrain[sub_key] = new["DLTrain"][sub_key]


    
system_config = Config()

config_file_path = os.path.join(current_path, "config.json")
with open(config_file_path, 'r') as f:
        system_config.update_config(json.load(f))