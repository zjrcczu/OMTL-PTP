import copy
import os
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import LinearSegmentedColormap
import datetime
# import cv2
from random import randint
import numpy as np
import json
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from models.model_memory_IRM import model_memory_IRM
import io
from PIL import Image
from torchvision.transforms import ToTensor
# import dataset_invariance
import index_qualitative
import tqdm
import os

import torch
from data.dataloader import data_generator
from utils.config import Config

import math


def rotate_traj(past, future):
    past_diff = past[:, 0]

    past_theta = torch.atan(torch.div(past_diff[:, 1], past_diff[:, 0] + 1e-5))
    past_theta = torch.where((past_diff[:, 0] < 0), past_theta + math.pi, past_theta)



    rotate_matrix = torch.zeros((past_theta.size(0), 2, 2)).to(past_theta.device)
    rotate_matrix[:, 0, 0] = torch.cos(past_theta)
    rotate_matrix[:, 0, 1] = torch.sin(past_theta)
    rotate_matrix[:, 1, 0] = - torch.sin(past_theta)
    rotate_matrix[:, 1, 1] = torch.cos(past_theta)

    past_after = torch.matmul(rotate_matrix, past.transpose(1, 2)).transpose(1, 2)
    future_after = torch.matmul(rotate_matrix, future.transpose(1, 2)).transpose(1, 2)

    return past_after, future_after,past_theta.to(past_theta.device)

def arotate_traj(past, pv):
    past_diff = past[:, 0]

    past_theta = -pv


    rotate_matrix = torch.zeros((past_theta.size(0), 2, 2)).to(past_theta.device)
    rotate_matrix[:, 0, 0] = torch.cos(past_theta)
    rotate_matrix[:, 0, 1] = torch.sin(past_theta)
    rotate_matrix[:, 1, 0] = - torch.sin(past_theta)
    rotate_matrix[:, 1, 1] = torch.cos(past_theta)

    past_after = torch.matmul(rotate_matrix, past.transpose(1, 2)).transpose(1, 2)


    return past_after
import logging


def best_traj(pred_fake, pred_gt):
    bs, _, traj_len, _ = pred_fake.size()

    b = pred_gt.unsqueeze(1).repeat(1, 20, 1, 1)
    dis = torch.sum((pred_fake - b) ** 2, dim=3)  # bs,20,12,2->bs,20,12
    dis = torch.sum(dis, dim=2)  # bs,20
    error, ind = torch.min(dis, dim=1) # bs
    a=[]
    for i in range(ind.shape[0]):
        a.append(pred_fake[i,ind[i]])
    a=torch.stack(a)



    return a


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
        self.pred_len = config.future_len
        # Write details to file
        self.best_ade = 100
        self.previous_memory_len = {'ETH': 0, "ST": 0, 'ZARA': 0}
        self.previous_traj_len = {'ETH': 0, "ST": 0, 'ZARA': 0}
        self.dataset_name = config.dataset_name
        self.dest_path = ""
        self.name_test = str(datetime.datetime.now())[:19]
        dataset_name = config.dataset_name
        self.nm = self.cfg.dataset
        print(self.cfg.dataset)
        self.folder_test = 'training/training_IRM/' + dataset_name + '_' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)

    def online_fit(self):
        """
        Iterative refinement model training. The function loops over the data in the training set max_epochs times.
        :return: None
        """
        config = self.config

        for param in self.mem_n2n.parameters():
            param.requires_grad = True

        self._train_online()

    def model_update(self, a1, a2, a3, a4, a5, model_eth, model_zara, model_st, model_4, model_ft):

        model = copy.deepcopy(model_eth)
        model_dict = model.state_dict()
        # for k in model_dict:
        # model_dict[k]=torch.zeros_like( model_dict[k])
        zara_dict = model_zara.state_dict()
        eth_dict = model_eth.state_dict()
        st_dict = model_st.state_dict()
        model_4_dict = model_4.state_dict()
        ft_dict = model_ft.state_dict()
        # alpha = 0.06 # np.exp(-0.05*2)

        for par1, par2, par3, par4, part in zip(eth_dict, zara_dict, st_dict, model_4_dict, ft_dict):
            model_dict[par1] = a1 * eth_dict[par1] + a2 * zara_dict[par2] + a3 * st_dict[par3] + a4 * model_4_dict[
                par4] + a5 * ft_dict[part]

        model.load_state_dict(model_dict)

        model.memory_past = a1 * model_eth.memory_past + a2 * model_zara.memory_past + a3 * model_st.memory_past + a4 * model_4.memory_past + a5 * model_ft.memory_past
        model.memory_fut = a1 * model_eth.memory_fut + a2 * model_zara.memory_fut + a3 * model_st.memory_fut + a4 * model_4.memory_fut + a5 * model_ft.memory_fut

        return model

    def _train_online(self):
        """
        Training loop over the dataset for an epoch
        :return: loss
        """
        config = self.config

        self.mem_n2n.train()
        loss_all = 0

        ade_48s = fde_48s = 0
        samples = 0
        dict_metrics = {}

        a = []
        f = []
        num = 5
        a1 = 1.0 / num
        a2 = 1.0 / num
        a3 = 1.0 / num
        a4 = 1.0 / num
        a5 = 1.0 / num

        self.mem_n2n_1 = torch.load('./pretrained_models/CLTPMAN/eth')
        self.mem_n2n_2 = torch.load('./pretrained_models/CLTPMAN/zara2')
        self.mem_n2n_3 = torch.load('./pretrained_models/CLTPMAN/hotel')
        self.mem_n2n_4 = torch.load('./pretrained_models/CLTPMAN/univ')

        self.mem_n2n_1.train()
        self.mem_n2n_2.train()
        self.mem_n2n_3.train()
        self.mem_n2n_4.train()



        self.mem_n2n_new = self.model_update(0.01, 0.005, 0.03, 0.1, 0.04, self.mem_n2n_1, self.mem_n2n_2,
                                             self.mem_n2n_3, self.mem_n2n_4, self.mem_n2n_1)
        self.mem_n2n_new.memory_past = torch.zeros(100, 48).cuda()
        self.mem_n2n_new.memory_fut = torch.zeros(100, 48).cuda()
        self.mem_n2n_new.train()

        for param in self.mem_n2n_new.parameters():
            param.requires_grad = True

        start = time.time()
        pic_cnt = 0
        while not self.test_generator.is_epoch_end():
            data = self.test_generator()


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
                    past_normalized, fut_normalized ,pp = rotate_traj(past_normalized, fut_normalized)
                all_traj = torch.cat((past_normalized, fut_normalized), dim=1)
                all_rel = torch.zeros_like(all_traj)
                all_rel[:, 1:] = all_traj[:, 1:] - all_traj[:, :-1]
                # print('dd',all_rel.shape)
                past_rel = all_rel[:, :8].reshape(-1, 8, 2)
                future_rel = all_rel[:, 8:].reshape(-1, 12, 2)
                past_normalized_one = past_normalized.reshape(-1, 8, 2)
                fut_normalized_one = fut_normalized.reshape(-1, 12, 2)

                 #(bs,2)
                with torch.no_grad():
                    pred_1 = self.mem_n2n_1(past_rel, obs_traj=past_normalized_one)
                    pred_1 = pred_1 * scale
                    pred=mse_loss(pred_1,fut_normalized_one)
                    pred_1=arotate_traj(pred,pp)+last_frame
                    '''pf = arotate_traj(fut_normalized , pp)+last_frame
                    print(pf-future)'''


                plt.figure(figsize=(20, 15), dpi=100)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                lll=pred_1.shape[0]
                ground_truth_input_x_piccoor = (
                    past[:, :, :].cpu().numpy()[:,:,  0]#.T
                )
                ground_truth_input_y_piccoor = (
                    past[:, :, :].cpu().numpy()[:, :, 1]#.T bs,8,2
                )
                ground_truth_output_x_piccoor = (
                    future[:,  :,:].cpu().numpy()[:, :, 0]#.T
                )
                ground_truth_output_y_piccoor = (
                    future[:, :, :].cpu().numpy()[:, :, 1]#.T
                )
                model_output_x_piccoor = (
                    pred_1[:, :,:].cpu().numpy()[:, :, 0]#.T
                )
                model_output_y_piccoor = (
                    pred_1[:, :, :].cpu().numpy()[:, :, 1]#.T
                )

                for i in range(ground_truth_output_x_piccoor.shape[0]):
                    ### plt 20 trajectories


                    observed_line = plt.plot(
                        ground_truth_input_x_piccoor[i, :],
                        ground_truth_input_y_piccoor[i, :],
                        "r-",
                        linewidth=4,
                        label="Observed Trajectory",
                    )[0]
                    observed_line.axes.annotate(
                        "",
                        xytext=(
                            ground_truth_input_x_piccoor[i, -2],
                            ground_truth_input_y_piccoor[i, -2],
                        ),
                        xy=(
                            ground_truth_input_x_piccoor[i, -1],
                            ground_truth_input_y_piccoor[i, -1],
                        ),
                        arrowprops=dict(
                            arrowstyle="->", color=observed_line.get_color(), lw=4
                        ),
                        size=20,
                    )
                    ground_line = plt.plot(
                        np.append(
                            ground_truth_input_x_piccoor[i, -1],
                            ground_truth_output_x_piccoor[i, :],
                        ),
                        np.append(
                            ground_truth_input_y_piccoor[i, -1],
                            ground_truth_output_y_piccoor[i, :],
                        ),
                        "b-",
                        linewidth=4,
                        label="Ground Truth",
                    )[0]


                    plt.scatter(ground_truth_input_x_piccoor[i, -1], ground_truth_input_y_piccoor[i, -1], s=300, c='b', marker='*')
                    predict_line = plt.plot(
                        np.append(
                            ground_truth_input_x_piccoor[i, -1],
                            model_output_x_piccoor[i, :],
                        ),
                        np.append(
                            ground_truth_input_y_piccoor[i, -1],
                            model_output_y_piccoor[i, :],
                        ),
                        color="g",  # ffff00
                        ls="--",
                        linewidth=4,
                        label="Predicted Trajectory",
                    )[0]

                # plt.axis("off")
                plt.savefig(
                    "./traj_fig/pic_{}.png".format(pic_cnt)
                )
                plt.close()
                pic_cnt += 1

        return 0


