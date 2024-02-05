import os
import matplotlib.pyplot as plt
import datetime
import io

# import thop
from PIL import Image
from torchvision.transforms import ToTensor
import json
import torch
import logging
import torch.nn as nn
#from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.model_encdec import model_encdec
import dataset_invariance
from torch.autograd import Variable
import tqdm


import argparse
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

class Trainer:
    def __init__(self, config):
        """
        The Trainer class handles the training procedure for training the autoencoder.
        param config: configuration parameters (see train_ae.py)
        """

        # test folder creating
        self.name_test = str(datetime.datetime.now())[:13]
        self.folder_tensorboard = 'runs/runs-ae/'
        self.folder_test = 'training/training_ae/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
        self.file = open(self.folder_test + "details.txt", "w")

        print('Creating dataset...')

        self.settings = {
            "batch_size": config.batch_size,
            "use_cuda": config.cuda,
            "dim_feature_tracklet": config.obs_len * 2,
            "dim_feature_future": config.pred_len * 2,
            "dim_embedding_key": config.dim_embedding_key,
            "past_len": config.obs_len,
            "future_len": config.pred_len,
        }
        self.max_epochs = config.max_epochs

        self.cfg = Config(config.cfg, config.info, config.tmp, create_dirs=True)

        self.log = open(os.path.join(self.cfg.log_dir, 'log.txt'), 'a+')
        self.train_generator = data_generator(self.cfg, self.log, split='train', phase='training')
        self.eval_generator = data_generator(self.cfg, self.log, split='val', phase='testing')
        self.test_generator = data_generator(self.cfg, self.log, split='test', phase='testing')
        # model
        self.mem_n2n = model_encdec(self.settings)
        # # thop统计模型大小
        # flops, params = thop.profile(self.mem_n2n.cuda(), inputs=(torch.randn(1, 8, 2).cuda(), torch.randn(1, 12, 2).cuda(), torch.randn(1, 8, 2).cuda()))
        # print('flops:', flops/1e9, 'G')
        # print('params:', params/1e6, 'M')

        # loss
        self.criterionLoss = nn.MSELoss()

        self.opt = torch.optim.Adam(self.mem_n2n.parameters(), lr=config.learning_rate)
        self.iterations = 0
        if config.cuda:
            self.criterionLoss = self.criterionLoss.cuda()
            self.mem_n2n = self.mem_n2n.cuda()
        self.start_epoch = 0
        self.config = config

        self.file.close()




    def fit(self):
        """
        Autoencoder training procedure. The function loops over the data in the training set max_epochs times.
        :return: None
        """
        config = self.config
        # Training loop
        best=100
        for epoch in range(self.start_epoch, config.max_epochs):

            print(' ----- Epoch: {}'.format(epoch))
            loss = self._train_single_epoch(epoch)
            print('Loss: {}'.format(loss))


            if (epoch + 0) % 2 == 0: # 20

                print('test on TEST dataset')

                dict_metrics_test = self.evaluate(epoch + 1)
                print('best_ade:',best,' accuracy_test/eucl_mean', dict_metrics_test['eucl_mean'], epoch)
                if best>dict_metrics_test['eucl_mean']:
                    best=dict_metrics_test['eucl_mean']
                # Save model checkpoint
                    torch.save(self.mem_n2n, self.folder_test + 'model_ae_epoch_' + str(epoch) + '_' + self.name_test)

        # Save final trained model
        #torch.save(self.mem_n2n, self.folder_test + 'model_ae_' + self.name_test)

    def evaluate(self, epoch=0):
        """
        Evaluate the model.
        :param loader: pytorch dataloader to loop over the data
        :param epoch: current epoch (default 0)
        :return: a dictionary with performance metrics
        """

        eucl_mean = horizon10s = horizon20s = horizon30s = horizon40s = 0
        dict_metrics = {}

        # Loop over samples
        data_len =0
        with torch.no_grad():
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
                all_rel[:, 1:] = all_traj[:, 1:] - all_traj[:, :19]

                ### calculate rel
                past_rel = all_rel[:, :8]
                future_rel = all_traj[:, 8:]

                pred = self.mem_n2n(past_rel, future_rel, past_normalized)
                pred = pred*scale

                data_len=data_len+past.size(0)
                distances = torch.norm(pred -fut_normalized, dim=2)
                eucl_mean += torch.sum(torch.mean(distances, 1))
                horizon10s += torch.sum(distances[:, 3])
                horizon20s += torch.sum(distances[:, 6])
                horizon30s += torch.sum(distances[:, 9])
                horizon40s += torch.sum(distances[:, 11])


        dict_metrics['eucl_mean'] = eucl_mean / data_len
        dict_metrics['horizon10s'] = horizon10s / data_len
        dict_metrics['horizon20s'] = horizon20s / data_len
        dict_metrics['horizon30s'] = horizon30s / data_len
        dict_metrics['horizon40s'] = horizon40s / data_len
        print('len: ',data_len)

        return dict_metrics

    def _train_single_epoch(self,epoch):
        """
        Training loop over the dataset for an epoch
        :return: loss
        """
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


                pred = self.mem_n2n(past_rel, future_rel, past_normalized)

                pred = pred * scale
                mse = torch.nn.MSELoss()
                l2_loss_sum_rel = mse(pred, fut_normalized)
                loss=torch.zeros(1).cuda()

                loss = loss + l2_loss_sum_rel
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mem_n2n.parameters(), 1.0, norm_type=2)
                self.opt.step()
                count=count+1
                loss_all=loss_all+loss
        print(epoch,'ll',loss_all/count)

        return loss.item()
