import copy
import os
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import LinearSegmentedColormap
import datetime
#import cv2
from random import randint
import numpy as np
import json
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.model_memory_IRM import model_memory_IRM
import io
from PIL import Image
from torchvision.transforms import ToTensor
#import dataset_invariance
import index_qualitative
import tqdm
import os

import torch
from data.dataloader import data_generator
from utils.config import Config

import math



def rotate_traj(past, future):
		past_diff = past[:, 0]
		past_theta = torch.atan(torch.div(past_diff[:, 1], past_diff[:, 0]+1e-5))
		past_theta = torch.where((past_diff[:, 0]<0), past_theta+math.pi, past_theta)

		rotate_matrix = torch.zeros((past_theta.size(0), 2, 2)).to(past_theta.device)
		rotate_matrix[:, 0, 0] = torch.cos(past_theta)
		rotate_matrix[:, 0, 1] = torch.sin(past_theta)
		rotate_matrix[:, 1, 0] = - torch.sin(past_theta)
		rotate_matrix[:, 1, 1] = torch.cos(past_theta)

		past_after = torch.matmul(rotate_matrix, past.transpose(1, 2)).transpose(1, 2)
		future_after = torch.matmul(rotate_matrix, future.transpose(1, 2)).transpose(1, 2)

		return past_after, future_after

import logging

class Trainer:
    def __init__(self, config):
        """
        Trainer class for training the Iterative Refinement Module (IRM)
        :param config: configuration parameters (see train_IRM.py)
        """

        self.index_qualitative = index_qualitative.dict_test
        self.name_run = 'runs/runs-IRM/'

        self.cfg = Config(config.cfg, config.info, config.tmp, create_dirs=True)

        self.log = open(os.path.join(self.cfg.log_dir, 'log.txt'), 'a+')
        self.train_generator = data_generator(self.cfg, self.log, split='train', phase='training')
        self.eval_generator = data_generator(self.cfg, self.log, split='val', phase='testing')
        self.test_generator = data_generator(self.cfg, self.log, split='test', phase='testing')



        self.num_prediction = config.preds
        self.settings = {
            "batch_size": config.batch_size,
            "use_cuda": config.cuda,
            "dim_embedding_key": config.dim_embedding_key,
            "num_prediction": self.num_prediction,
            "past_len": config.past_len,
            "future_len": config.future_len,
            "th": config.th
        }
        self.max_epochs = config.max_epochs

        # load pretrained model and create memory_model
        self.model = torch.load(config.model)
        self.mem_n2n = model_memory_IRM(self.settings, self.model)
        self.mem_n2n.past_len = config.past_len
        self.mem_n2n.future_len = config.future_len

        self.criterionLoss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.mem_n2n.parameters(), lr=config.learning_rate)
        self.iterations = 0
        if config.cuda:
            self.criterionLoss = self.criterionLoss.cuda()
            self.mem_n2n = self.mem_n2n.cuda()
        self.start_epoch = 0
        self.config = config
        self.pred_len=config.future_len
        # Write details to file
        self.best_ade = 100
        self.previous_memory_len = {'ETH':0,"ST":0,'ZARA':0}
        self.previous_traj_len = {'ETH':0,"ST":0,'ZARA':0}
        self.dataset_name = 0
        self.dest_path =""
        self.name_test = str(datetime.datetime.now())[:19]
        dataset_name=config.dataset_name
        self.folder_test = 'training/training_IRM/' + dataset_name + '_' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)

    def fit(self):
        """
        Iterative refinement model training. The function loops over the data in the training set max_epochs times.
        :return: None
        """
        config = self.config

        # freeze autoencoder layers
        for param in self.mem_n2n.conv_past.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.encoder_past.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.conv_fut.parameters():
            param.requires_grad = False

        for param in self.mem_n2n.encoder_fut.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.linear_controller.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.decoder.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.FC_output.parameters():
            param.requires_grad = False

        for param in self.mem_n2n.decoder_fut.parameters():
            param.requires_grad = True
        for param in self.mem_n2n.FC_output_fut.parameters():
            param.requires_grad = True

        for param in self.mem_n2n.Lin_Q.parameters():
            param.requires_grad = True
        for param in self.mem_n2n.Lin_K.parameters():
            param.requires_grad = True
        for param in self.mem_n2n.Lin_V.parameters():
            param.requires_grad = True
        for param in self.mem_n2n.Lin_Sgmoid.parameters():
            param.requires_grad = True#True

        start = time.time()

        end = time.time()
        print('writing time: ' + str(end-start))
        ###########################
        a=torch.load('./pretrained_models/model_controller/hotel_100_mem.pth')
        print(a.shape)
        self.mem_n2n.memory_past=a[:,:48]
        self.mem_n2n.memory_fut=a[:,48:]

        for epoch in range(self.start_epoch, config.max_epochs):
            self.mem_n2n.train()

            print('epoch: ' + str(epoch))
            start = time.time()
            loss = self._train_single_epoch(epoch)
            end = time.time()
            print('Epoch took: {} Loss: {}'.format(end - start, loss))

            #if (epoch + 1) in step_results:
            # Test model while training
            print('start test')
            start_test = time.time()

            dict_metrics_test = self.evaluate( epoch + 1)  # self.testloader
            #dict_metrics_val= self.evaluate(val_loader, epoch + 1)#self.testloader
            end_test = time.time()
            print('Test took: {}'.format(end_test - start_test))

            # Save model checkpoint
            if dict_metrics_test['euclMean'].item()<self.best_ade:
                a=[dict_metrics_test['euclMean'],dict_metrics_test['horizon40s']]
                print('ade:',dict_metrics_test['euclMean'],'fde:',dict_metrics_test['horizon40s'])
                self.best_ade=dict_metrics_test['euclMean'].item()

                torch.save(self.mem_n2n, self.folder_test + '/model_IRM_epoch_' + str(epoch) + '_' + self.name_test)
                #self.save_results('val',dict_metrics_val, epoch=epoch + 1)
            print(a)
            #print('val ade:', dict_metrics_val['euclMean'], 'fde:', dict_metrics_val['horizon40s'])
        # Save final trained model
        torch.save(self.mem_n2n, self.folder_test+ '/' + 'model_mantra_' + self.name_test)



    def evaluate(self,epoch=0):
        """
        Evaluate model. Future trajectories are predicted and
        :param loader: data loader for testing data
        :param epoch: epoch index (default: 0)
        :return: dictionary of performance metrics
        """

        self.mem_n2n.eval()
        data_len=0
        ade_48s = fde_48s = 0
        samples = 0
        dict_metrics = {}

        with torch.no_grad():
            while not self.test_generator.is_epoch_end():

                data = self.test_generator()
                if data is not None:

                    past  = torch.stack(data['pre_motion_3D']).cuda()
                    future = torch.stack(data['fut_motion_3D']).cuda()
                    last_frame = past[:, -1:]
                    past_normalized =  past - last_frame
                    fut_normalized  =  future - last_frame

                    scale = 1
                    if self.cfg.scale.use:
                        scale = torch.mean(torch.norm(past_normalized[:, 0], dim=1)) / 3
                        if scale < self.cfg.scale.threshold:
                            scale = 1
                        else:
                            if self.cfg.scale.type == 'divide':
                                scale = scale / self.cfg.scale.large
                            elif self.cfg.scale.type == 'minus':
                                scale = scale - self.cfg.scale.large
                        if self.cfg.scale.type == 'constant':
                            scale = self.cfg.scale.value
                        past_normalized = past_normalized / scale

                    if self.cfg.rotation:
                        past_normalized, fut_normalized = rotate_traj(past_normalized, fut_normalized)
                    all_traj = torch.cat((past_normalized, fut_normalized), dim=1)
                    all_rel = torch.zeros_like(all_traj)
                    all_rel[:, 1:] = all_traj[:, 1:] - all_traj[:, :19]

                    ### calculate rel
                    past_rel = all_rel[:, :8]
                    future_rel = all_traj[:, 8:]

                    pred = self.mem_n2n(past_rel,obs_traj=past_normalized)

                    pred = pred * scale
                    future_rep = fut_normalized.unsqueeze(1).repeat(1, 20, 1, 1)
                    distances = torch.norm(pred- future_rep, dim=3)
                    distances = torch.where(torch.isnan(distances), torch.full_like(distances, 10), distances)
                    # N, K, T

                    mean_distances = torch.mean(distances[:, :, -1:], dim=2)
                    mean_distances_ade = torch.mean(distances, dim=2)

                    index_min = torch.argmin(mean_distances, dim=1)
                    min_distances = distances[torch.arange(0, len(index_min)), index_min]

                    index_min_ade = torch.argmin(mean_distances_ade, dim=1)
                    min_distances_ade = distances[torch.arange(0, len(index_min_ade)), index_min_ade]

                    fde_48s += torch.sum(min_distances[:, -1])
                    ade_48s += torch.sum(torch.mean(min_distances_ade, dim=1))

                    samples += distances.shape[0]

            dict_metrics['fde_48s'] = fde_48s / samples
            dict_metrics['ade_48s'] = ade_48s / samples

            dict_metrics['euclMean']    = dict_metrics['ade_48s']# torch.tensor(ade1,dtype=torch.float)#ade.avg
            dict_metrics['horizon40s'] = dict_metrics['fde_48s']#torch.tensor(de1,dtype=torch.float)

        return dict_metrics

    def _train_single_epoch(self,epoch):
        """
        Training loop over the dataset for an epoch
        :return: loss
        """
        config = self.config
        self.mem_n2n.train()
        count=0
        loss_all=0
        while not self.train_generator.is_epoch_end():
            data = self.train_generator()
            if data is not None:

                past = torch.stack(data['pre_motion_3D']).cuda()
                future = torch.stack(data['fut_motion_3D']).cuda()
                last_frame = past[:, -1:]
                past_normalized = past - last_frame
                fut_normalized = future - last_frame

                scale = 1
                if self.cfg.scale.use:
                    scale = torch.mean(torch.norm(past_normalized[:, 0], dim=1)) / 3
                    if scale < self.cfg.scale.threshold:
                        scale = 1
                    else:
                        if self.cfg.scale.type == 'divide':
                            scale = scale / self.cfg.scale.large
                        elif self.cfg.scale.type == 'minus':
                            scale = scale - self.cfg.scale.large
                    if self.cfg.scale.type == 'constant':
                         scale = self.cfg.scale.value
                    past_normalized = past_normalized / scale

                if self.cfg.rotation:
                    past_normalized, fut_normalized = rotate_traj(past_normalized, fut_normalized)
                all_traj = torch.cat((past_normalized, fut_normalized), dim=1)
                all_rel = torch.zeros_like(all_traj)
                all_rel[:, 1:] = all_traj[:, 1:] - all_traj[:, :-1]

                ### calculate rel
                past_rel = all_rel[:, :8]
                future_rel = all_traj[:, 8:]

                self.opt.zero_grad()

                output = self.mem_n2n(past_rel,obs_traj=past_normalized)

                output = output* scale
                b = fut_normalized.unsqueeze(1).repeat(1, 20, 1, 1)
                dis = torch.sum((output - b)**2, dim=3)
                dis = torch.sum(dis, dim=2)
                error, ind = torch.min(dis, dim=1)

                loss =torch.zeros(1).cuda()
                loss=loss+torch.mean(error)/12.0

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mem_n2n.parameters(), 1.0, norm_type=2)
                self.opt.step()
                count = count + b.size(0)
                loss_all = loss_all + loss

        print(epoch, 'll', loss_all / count)

        return loss.item()


