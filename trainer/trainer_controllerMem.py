import os
import matplotlib.pylab as pl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import datetime
import numpy as np
import cv2
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.model_controllerMem import model_controllerMem
import dataset_invariance
from torch.autograd import Variable
import io
from PIL import Image
from torchvision.transforms import ToTensor
import time
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


class Trainer():
    def __init__(self, config):
        """
        The Trainer class handles the training procedure for training the memory writing controller.
        :param config: configuration parameters (see train_controllerMem.py)
        """

        self.name_test = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.folder_tensorboard = 'runs/runs-createMem/'
        self.folder_test = 'training/training_controller/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
        self.file = open(self.folder_test + "details.txt", "w")

        print('creating dataset...')
        tracks = json.load(open(config.dataset_file))
        self.dim_clip = 180
        self.pred_len=config.future_len

        self.cfg = Config(config.cfg, config.info, config.tmp, create_dirs=True)

        self.log = open(os.path.join(self.cfg.log_dir, 'log.txt'), 'a+')
        self.train_generator = data_generator(self.cfg, self.log, split='train', phase='training')
        self.eval_generator = data_generator(self.cfg, self.log, split='val', phase='testing')
        self.test_generator = data_generator(self.cfg, self.log, split='test', phase='testing')


        print('dataset created')
        self.settings = {
            "batch_size": config.batch_size,
            "use_cuda": config.cuda,
            "dim_embedding_key": config.dim_embedding_key,
            "num_prediction": config.best_k,
            "past_len": config.past_len,
            "future_len": config.future_len,
            "th": config.th
        }
        self.max_epochs = config.max_epochs
        # load pretrained model and create memory model
        self.model_ae = torch.load(config.model_ae)
        self.mem_n2n = model_controllerMem(self.settings, self.model_ae)
        self.mem_n2n.future_len = config.future_len
        self.mem_n2n.past_len = config.past_len

        self.opt = torch.optim.Adam(self.mem_n2n.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, 0.5)
        self.iterations = 0
        if config.cuda:
            self.mem_n2n = self.mem_n2n.cuda()
        self.start_epoch = 0
        self.config = config

        # Write details to file

        self.file.close()
        self.best_ade=100

        self.mem_past = 0
        self.mem_fut = 0



    def fit(self):
        """
        Writing controller training. The function loops over the data in the training set max_epochs times.
        :return: None
        """
        config = self.config

        # freeze autoencoder layers
        for param in self.mem_n2n.conv_past.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.conv_fut.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.encoder_past.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.encoder_fut.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.decoder.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.FC_output.parameters():
            param.requires_grad = False

        for param in self.mem_n2n.linear_controller.parameters():
            param.requires_grad = True

        # Memory Initialization
        self.mem_n2n.init_memory(0)####?
        self.mem_past=self.mem_n2n.memory_past
        self.mem_fut=self.mem_n2n.memory_fut


        from torch.nn import init


        init.uniform_( self.mem_n2n.linear_controller.weight, a=0.99, b=1.0)
        init.constant_(self.mem_n2n.linear_controller.bias, val=0.0)
        print(self.mem_n2n.linear_controller.weight)
        # Main training loop
        for epoch in range(self.start_epoch, config.max_epochs):

            #self.mem_n2n.init_memory(self.data_train)############
            self.mem_n2n.memory_past = self.mem_past
            self.mem_n2n.memory_fut = self.mem_fut
            self.mem_n2n.pred_gt = torch.zeros((21, 12, 2)).cuda()
            self.mem_n2n.past_gt = torch.zeros((21, 8, 2)).cuda()
            self.mem_n2n.pred_rel_gt = torch.zeros((21, 12, 2)).cuda()
            self.mem_n2n.past_rel_gt = torch.zeros((21, 8, 2)).cuda()

            print('epoch: ' + str(epoch))
            start = time.time()
            loss = self._train_single_epoch()
            end = time.time()

            print('Epoch took: {} Loss: {}'.format(end - start, loss))
            #self.save_plot_controller(epoch)

            #if (epoch + 1) % 5 != 0:
            # Test model while training
            print('start test')
            start_test = time.time()
            dict_metrics_test,flag = self.evaluate( epoch + 1)#self.test_loader
            end_test = time.time()
            print('Test took: {}'.format(end_test - start_test))

            # Tensorboard summary: test
            #self.writer.add_scalar('accuracy_test/ade', dict_metrics_test['eucl_mean'], epoch)
            #self.writer.add_scalar('accuracy_test/fde', dict_metrics_test['horizon40s'], epoch)


            # print memory on tensorboard
            mem_size = self.mem_n2n.memory_past.shape[0]

            '''for i in range(mem_size):
                track_mem = self.mem_n2n.check_memory(i).squeeze(0).cpu().detach().numpy()
                plt.plot(track_mem[:, 0], track_mem[:, 1], marker='o', markersize=1)
            plt.axis('equal')
            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            image = Image.open(buf)
            image = ToTensor()(image).unsqueeze(0)
            self.writer.add_image('memory_content/memory', image.squeeze(0), epoch)
            plt.close()'''
            # Save model checkpoint
            if flag:
                torch.save(self.mem_n2n,
                           self.folder_test + 'model_controller_best_' + str(epoch) + '_' + self.name_test)


        # Save final trained model
        torch.save(self.mem_n2n, self.folder_test + 'model_controller_' + self.name_test)




    def evaluate(self,epoch=0):
        """
        Evaluate model. Future trajectories are predicted and
        :param loader: data loader for testing data
        :param epoch: epoch index (default: 0)
        :return: dictionary of performance metrics
        """
        self._memory_writing()
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

                    pred = self.mem_n2n(past_rel, future_rel, past_normalized)
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


            if self.best_ade > dict_metrics['ade_48s']:
                self.best_ade=dict_metrics['ade_48s']
                flag=True
            else:flag=False
            print('best:',self.best_ade,"memory size: " + str(self.mem_n2n.memory_past.size()) )
            #print('pred:',self.mem_n2n.pred_rel_gt.shape)
            print(" *  ADE  :",dict_metrics['ade_48s']," FDE  :",dict_metrics['fde_48s'])

        return dict_metrics,flag

    def _train_single_epoch(self):
        """
        Training loop over the dataset for an epoch
        :return: loss
        """
        data_len=0
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
                prob, sim = self.mem_n2n(past_rel, future_rel,obs_traj=past_normalized,pred_gt=fut_normalized,scale=scale)
                loss = self.ControllerLoss(prob, sim) # (bs,1)
                loss.backward()
                self.opt.step()

        return loss.item()

    def ControllerLoss(self, prob, sim):
        """
        Loss to train writing controller:
        :param prob: writing probability generated by controller
        :param sim: similarity (between 0 and 1) between better prediction and ground-truth.
        :return: loss
        """
        loss = prob * (1 - sim) + (1 - prob)* sim  # prob * sim + (1 - prob) * (1 - sim) #

        return sum(loss)

    def _memory_writing(self):
        """
        writing in the memory with controller (loop over all train dataset)
        :return: loss
        """
        self.mem_n2n.init_memory(0)###############????
        #self.mem_n2n.memory_past = self.mem_past
        #self.mem_n2n.memory_fut = self.mem_fut
        self.mem_n2n.pred_gt = torch.zeros((21,12,2)).cuda()
        self.mem_n2n.past_gt = torch.zeros((21, 8, 2)).cuda()
        self.mem_n2n.pred_rel_gt = torch.zeros((21, 12, 2)).cuda()
        self.mem_n2n.past_rel_gt= torch.zeros((21, 8, 2)).cuda()


        self.mem_n2n.past_rel_gt = torch.zeros((21, 8, 2)).cuda()
        self.mem_n2n.pred_rel_gt = torch.zeros((21, 12, 2)).cuda()
        config = self.config
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
                    all_rel[:, 1:] = all_traj[:, 1:] - all_traj[:, :-1]

                    ### calculate rel
                    past_rel = all_rel[:, :8]
                    future_rel = all_traj[:, 8:]

                    _, _ = self.mem_n2n(past_rel, future_rel,obs_traj=past_normalized,pred_gt=fut_normalized,scale=scale)
            # save memory
        torch.save(self.mem_n2n.memory_past, self.folder_test + 'ETH_memory_past.pt')
        torch.save(self.mem_n2n.memory_fut, self.folder_test + 'ETH_memory_fut.pt')
        #print(self.mem_n2n.memory_fut.shape)

    def load(self, directory):
        pass
