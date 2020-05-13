from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Logger import system_log
from config import system_config

from models.model import create_model, save_model, load_model
from models.losses import CenterNetLoss
from utils.utils import AverageMeter
from dataset.dataLoader import dataset_loader

import torch 
import math
import time
import os 
import numpy as np 

this_dir = os.path.dirname(__file__)

class Trainer(object):
    def __init__(self, model_name, train_folder, validation_folder=None):
        self.model = create_model(model_name).cuda()
        if system_config.DLTrain['train_with_multi_gpu']:
            self.model = torch.nn.DataParallel(self.model)

        self.train_folder = train_folder
        self.val_folder = validation_folder
        self.loss = CenterNetLoss()
        self.loss_states = ["loss", "hm_loss", "wh_loss", "off_loss"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=system_config.DLTrain['lr'])

        self.model_root = os.path.join(this_dir, 'ckpt')
        self.model_root = os.path.join(self.model_root, model_name)
        if not os.path.exists(self.model_root):
            os.makedirs(self.model_root)

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate"""
        lr = system_config.DLTrain['lr']
        cycle = system_config.DLTrain['lr_cycle']
        
        c = (np.cos(((epoch%cycle)*2.0*math.pi)/cycle)+1) / 2
        lr = (lr * c) + 1e-6
                
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        system_log.WriteLine(f"update learning rate to {lr}")
        
    def run(self):
        max_epoch = system_config.DLTrain['max_epoch']

        # data loader
        train_loader = torch.utils.data.DataLoader(dataset_loader(self.train_folder), \
                                                   batch_size=system_config.DLTrain['batch_size'], shuffle = True, num_workers=10, pin_memory=True)
        if self.val_folder is not None:
            val_loader = torch.utils.data.DataLoader(dataset_loader(self.val_folder), \
                                                     batch_size=system_config.DLTrain['batch_size'], shuffle = True, num_workers=10, pin_memory=True)

        # train
        total_loss = AverageMeter()
        total_start = time.time()
        min_loss = np.inf
        min_val_loss = np.inf 

        for epoch_idx in range(1, max_epoch):
            self.adjust_learning_rate(self.optimizer, epoch_idx)
            
            avg_loss_stats, epoch_time = self.run_epoch(self.model, "train", epoch_idx, max_epoch, train_loader)
            total_loss.update(avg_loss_stats["loss"].avg)
            system_log.WriteLine(f"Epoch {epoch_idx} train finish, total loss: {total_loss.avg:.8f}, epoch cost time: {epoch_time:.8f}sec!")

            # save ckpt
            ckpt_path = os.path.join(self.model_root, f"ckpt_{epoch_idx}.pth")
            save_model(ckpt_path, epoch_idx, self.model)

            # save min loss
            if min_loss > avg_loss_stats['loss'].avg:
                min_loss = avg_loss_stats['loss'].avg 
                ckpt_minloss_path = os.path.join(self.model_root, f"ckpt_minloss.pth")
                save_model(ckpt_minloss_path, epoch_idx, self.model)

            # validation
            if self.val_folder is not None and epoch_idx % system_config.DLTrain['validation_per_epoch'] == 0:
                system_log.WriteLine(f"epoch_idx is {epoch_idx}, run validation...")
                val_epoch_time, val_avg_loss_stats = self.run_epoch(self.model, "val", epoch_idx, max_epoch, val_loader)
                system_log.WriteLine(f"validation finish, total loss is {val_avg_loss_stats['loss'].avg:.8f}, cost time: {val_epoch_time:.8f}sec!")

                if min_val_loss > val_avg_loss_stats['loss'].avg:
                    min_val_loss = val_avg_loss_stats['loss'].avg 
                    ckpt_best_path = os.path.join(self.model_root, f"ckpt_best.pth")
                    save_model(ckpt_best_path, epoch_idx, self.model)

        total_end = time.time()
        total_time = total_end - total_start
        system_log.WriteLine(f"all train finish, total epoch:{max_epoch}, total loss:{total_loss.avg:.8f}, cost time:{total_time:.8f}sec!")

    def run_epoch(self, model, phase, epoch_idx, max_epoch, data_loader):
        if phase == "train":
            model.train()
        else:
            model.eval()
            torch.cuda.empty_cache()

        avg_loss_stats = {l: AverageMeter() for l in self.loss_states}
        start = time.time()
        for batch in data_loader:
            batch = {k: v.cuda() for k,v in batch.items()}
            outputs = self.model(batch["input"])
            loss, loss_stats = self.loss(outputs, batch)
        
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            for k in self.loss_states:
                avg_loss_stats[k].update(float(loss_stats[k]))
    
            if phase == "train":
                system_log.WriteLine(f"Epoch [{epoch_idx}/{max_epoch}]: "
                                     f"loss: {avg_loss_stats['loss'].avg:.8f},  "
                                     f"hm_loss: {avg_loss_stats['hm_loss'].avg:.8f},  "
                                     f"wh_loss: {avg_loss_stats['wh_loss'].avg:.8f},  "
                                     f"off_loss: {avg_loss_stats['off_loss'].avg:.8f} ")

        end = time.time()

        epoch_time = end - start

        return avg_loss_stats, epoch_time

    def resume(self, model_path):
        self.model = load_model(self.model, model_path)

            

        

        

