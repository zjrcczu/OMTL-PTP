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


from collections import OrderedDict
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

		return past_after, future_after,past_theta

import logging
def mse_loss(pred_fake, pred_gt):
    bs,_,traj_len,_=pred_fake.size()

    b = pred_gt.unsqueeze(1).repeat(1, 20, 1, 1)
    dis = torch.sum((pred_fake - b) ** 2, dim=3)  # bs,20,12,2->bs,20,12
    dis = torch.sum(dis, dim=2)  # bs,20
    error, ind = torch.min(dis, dim=1)  # bs

    loss = torch.zeros(1).cuda()
    loss = loss + torch.mean(error) / traj_len
    return loss
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
        self.pred_len=config.future_len
        # Write details to file
        self.best_ade = 100
        self.previous_memory_len = {'ETH':0,"ST":0,'ZARA':0}
        self.previous_traj_len = {'ETH':0,"ST":0,'ZARA':0}
        self.dataset_name = config.dataset_name
        self.dest_path =""
        self.name_test = str(datetime.datetime.now())[:19]
        dataset_name=config.dataset_name
        self.nm=self.cfg.dataset
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


        self._train_online()




    def model_update(self, a1, a2, a3, a4, a5,model_1, model_2, model_3,model_4, model_ft):


        model = copy.deepcopy(model_ft)# t1 test 3


        model_dict = dict()#model.state_dict()
        m1_dict = model_1.state_dict()
        m2_dict = model_2.state_dict()
        m3_dict = model_3.state_dict()
        m4_dict = model_4.state_dict()
        ft_dict = model_ft.state_dict()

        # alpha = 0.06 # np.exp(-0.05*2)


        for par1, par2, par3, par4, part in zip( m1_dict,m2_dict, m3_dict,m4_dict, ft_dict):

            vl1 = a1 * torch.tensor(m1_dict[par1],dtype=torch.float64).cpu().detach()
            vl2 = a2 *  torch.tensor(m2_dict[par2],dtype=torch.float64).cpu().detach()
            vl3 = a3 *  torch.tensor(m3_dict[par3],dtype=torch.float64).cpu().detach()
            vl4 = a4 *  torch.tensor(m4_dict[par4],dtype=torch.float64).cpu().detach()
            vl5 = a5 *  torch.tensor(ft_dict[part],dtype=torch.float64).cpu().detach()





            k= vl1+vl2+vl3+vl4+vl5



            s=copy.deepcopy(k.detach())



            model_dict[par1]=s


        model.load_state_dict(model_dict,strict=False)
        for p in model.parameters():    # t1,0 k0,1 test  3,0,2,1
            p.requires_grad = True
        '''for par1, par2, par3, par4, part in zip(m1_dict, m2_dict, m3_dict,m4_dict, ft_dict):
              c=a1 * m1_dict[par1].detach()  + a2 *m2_dict[par2].detach()  + a3 * m3_dict[par3].detach()  + a4 * m4_dict[par4].detach() + a5*ft_dict[part].detach()
              s=copy.deepcopy(c)
              print(-model_dict[par1]+s  )   '''
        m1p = torch.tensor(model_1.memory_past, dtype=torch.float64).cuda().detach()
        m1f = torch.tensor(model_1.memory_fut, dtype=torch.float64).cuda().detach()
        model.memory_past = a1 * m1p + a2  * model_2.memory_past + a3  * model_3.memory_past +  a4 * model_4.memory_past+a5  * model_ft.memory_past
        model.memory_fut = a1 * m1f + a2  * model_2.memory_fut + a3  * model_3.memory_fut +  a4  * model_4.memory_fut+a5 * model_ft.memory_fut
        model.memory_past =torch.tensor(model.memory_past, dtype=torch.float32).cuda().detach()
        model.memory_fut =torch.tensor(model.memory_fut, dtype=torch.float32).cuda().detach()
        model.memory_fut.requires_grad = False
        model.memory_past.requires_grad = False
        return model
    def model_init(self, model_eth):

        model = copy.deepcopy(model_eth)
        model_dict = model.state_dict()


        for par1 in model_dict:
            model_dict[par1] = torch.zeros_like(model_dict[par1].detach())

        model.load_state_dict(model_dict,strict=False)


        #model.memory_past = a1 * model_eth.memory_past + a2 * model_zara.memory_past + a3 * model_st.memory_past + a4 * model_4.memory_past + a5 * model_ft.memory_past
        #model.memory_fut = a1 * model_eth.memory_fut + a2 * model_zara.memory_fut + a3 * model_st.memory_fut + a4 * model_4.memory_fut + a5 * model_ft.memory_fut

        return model


    def _train_online(self):
        """
        Training loop over the dataset for an epoch
        :return: loss
        """
        config = self.config

        self.mem_n2n.train()
        loss_all=0

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
        data_name='hotel'
        data_all=['eth','hotel','univ','zara1','zara2']
        data_all.remove(data_name)
        c=data_all
        print(c)
        vb=[0, 1, 2, 3]
        self.mem_n2n_1 = torch.load('./pretrained_models/CLTPMAN/'+str(c[vb[0]]))
        self.mem_n2n_2 = torch.load('./pretrained_models/CLTPMAN/'+str(c[vb[1]]))
        self.mem_n2n_3 = torch.load('./pretrained_models/CLTPMAN/'+str(c[vb[2]]))
        self.mem_n2n_4 = torch.load('./pretrained_models/CLTPMAN/'+str(c[vb[3]]))

        for param in self.mem_n2n_1.parameters():    # t1,0 k0,1 test  3,0,2,1
            param.requires_grad = False
        for param in self.mem_n2n_2.parameters():
            param.requires_grad = False
        for param in self.mem_n2n_3.parameters():
            param.requires_grad = False
        for param in self.mem_n2n_4.parameters():
            param.requires_grad = False

        self.mem_n2n_new = self.model_init(self.mem_n2n_2)# self.model_update(0.01, 0.005, 0.03,0.1,0.04, self.mem_n2n_1, self.mem_n2n_2,

        self.mem_n2n_new.memory_past = torch.zeros(100,48).cuda()
        self.mem_n2n_new.memory_fut  = torch.zeros(100,48).cuda()
        self.mem_n2n_new.memory_fut.requires_grad=False
        self.mem_n2n_new.memory_past.requires_grad = False
        self.mem_n2n_new.train()
        self.mem_n2n_1.eval()
        self.mem_n2n_2.eval()
        self.mem_n2n_3.eval()
        self.mem_n2n_4.eval()
        for param in self.mem_n2n_new.parameters():
            param.requires_grad = True


        start=time.time()

        aa= {'traj':[],'pred':[]}
        while not self.test_generator.is_epoch_end():
            data = self.test_generator()


            if data is not None:

                past = torch.stack(data['pre_motion_3D']).cuda()
                future = torch.stack(data['fut_motion_3D']).cuda()

                traj=torch.cat((past,future),1)
                aa['traj'].append(traj)

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
                    past_normalized, fut_normalized,past_theta = rotate_traj(past_normalized, fut_normalized)
                all_traj = torch.cat((past_normalized, fut_normalized), dim=1)
                all_rel = torch.zeros_like(all_traj)
                all_rel[:, 1:] = all_traj[:, 1:] - all_traj[:, :-1]
                #print('dd',all_rel.shape)

                for i in range(len(all_rel)):#len(all_rel)

                    ### calculate rel
                    past_rel = all_rel[i, :8].reshape(1,8,2)
                    future_rel = all_rel[i, 8:].reshape(1,12,2)
                    past_normalized_one=past_normalized[i].reshape(1,8,2)
                    fut_normalized_one=fut_normalized[i].reshape(1,12,2)
                    memn2n_copy = copy.deepcopy(self.mem_n2n_new)

                    with torch.no_grad():

                        pred_1 = self.mem_n2n_1(past_rel, obs_traj=past_normalized_one)
                        pred_1 = pred_1 * scale
                        mse_1 = mse_loss(pred_1,fut_normalized_one)


                        pred_2 = self.mem_n2n_2(past_rel, obs_traj=past_normalized_one)
                        pred_2 = pred_2* scale
                        mse_2 = mse_loss(pred_2,fut_normalized_one)

                        pred_3 = self.mem_n2n_3(past_rel, obs_traj=past_normalized_one)
                        pred_3 = pred_3 * scale
                        mse_3 = mse_loss(pred_3, fut_normalized_one)

                        pred_4 = self.mem_n2n_4(past_rel, obs_traj=past_normalized_one)
                        pred_4 = pred_4 * scale
                        mse_4 = mse_loss(pred_4,fut_normalized_one)

                        outputr = memn2n_copy(past_rel, obs_traj=past_normalized_one)
                        outputr = outputr * scale

                        future_rep = fut_normalized_one.unsqueeze(1).repeat(1, 20, 1, 1)
                        distances = torch.norm(outputr - future_rep, dim=3)
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

                        a.append(ade_48s.cpu().detach().numpy() / samples)
                        f.append(fde_48s.cpu().detach().numpy() / samples)



                    with torch.no_grad():
                        memn2n_ft = copy.deepcopy(memn2n_copy)
                        output1 = memn2n_ft(past_rel, obs_traj=past_normalized_one)
                        output1 = output1 * scale
                        mse_t = mse_loss(output1, fut_normalized_one)
                        predr = best_traj(output1, fut_normalized_one)
                        # print('h',predr.shape,past_theta.shape)
                        pred_best = arotate_traj(predr, past_theta[i].reshape(-1)) + last_frame[i].reshape(-1,2)
                        # print(pred_best.shape)
                        aa['pred'].append(pred_best)


                        ######
                        print(mse_1.cpu().detach().numpy(),mse_2.cpu().detach().numpy(),mse_3.cpu().detach().numpy(),mse_4.cpu().detach().numpy(),mse_t.cpu().detach().numpy())
                        beta = 0.5 # 0.28 0.5 1.18 k 0.3

                        a1 = a1 * math.pow(beta, mse_1)
                        a2 = a2 * math.pow(beta, mse_2)
                        a3 = a3 * math.pow(beta, mse_3)
                        a4 = a4 * math.pow(beta, mse_4)
                        a5 = a5 * math.pow(beta, mse_t)
                        al=a1+a2+a3+a4+a5
                        a1=a1/al
                        a2=a2/al
                        a3=a3/al
                        a4=a4/al
                        a5=a5/al

                        amin = min(a1, a2, a3, a4,a5)
                        amax = max(a1, a2, a3, a4,a5)
                        a1 = (a1 - amin) / (amax - amin)
                        a2 = (a2 - amin) / (amax - amin)
                        a3 = (a3 - amin) / (amax - amin)
                        a4 = (a4 - amin) / (amax - amin)
                        a5 = (a5 - amin) / (amax - amin)
                        s = 0.00001
                        a1 = max(a1, s)
                        a2 = max(a2, s)
                        a3 = max(a3, s)
                        a4 = max(a4, s)
                        a5 = max(a5, s)

                        # else:

                        a_all = a1 + a2 + a3 + a4 + a5

                        a1 = a1 / a_all
                        a2 = a2 / a_all
                        a3 = a3 / a_all
                        a4 = a4 / a_all
                        a5 = a5 / a_all

                        self.mem_n2n_new = self.model_update(a1,a2, a3,a4,a5, self.mem_n2n_1, self.mem_n2n_2,self.mem_n2n_3, self.mem_n2n_4, memn2n_ft)

                    '''self.mem_n2n1 = torch.load('./pretrained_models/CLTPMAN/hotel')
                    self.mem_n2n2 = torch.load('./pretrained_models/CLTPMAN/univ')
                    self.mem_n2n3 = torch.load('./pretrained_models/CLTPMAN/zara1')
                    self.mem_n2n4 = torch.load('./pretrained_models/CLTPMAN/zara2')
                    s=self.mem_n2n2.state_dict()
                    ss = self.mem_n2n_new.state_dict()
                    l=0
                    for p,pp in zip(s,ss):
                        l=l+torch.sum(s[p]-ss[p])
                    print(l)'''

                    print(samples,' ff', a1, a2, a3,a4,a5)
        aa['traj']=torch.cat(aa['traj'])
        aa['pred'] = torch.cat(aa['pred'])
        print(aa['traj'].shape,aa['pred'].shape)
        #torch.save(aa,'zara2.pt')




        dict_metrics['fde_48s'] = fde_48s / samples
        dict_metrics['ade_48s'] = ade_48s / samples

        end = time.time()
        print('time: ', end - start)
        import numpy as np
        import matplotlib.pyplot as plt

        x = np.arange(1, samples + 1)

        plt.plot(x, np.array(a)[:])
        plt.plot(x, np.array(f)[:])
        ade_ = torch.from_numpy(np.array(a))
        fde_ = torch.from_numpy(np.array(f))
        ade_fde = torch.cat((ade_.reshape(-1, 1), fde_.reshape(-1, 1)), 1)
        print(ade_fde.shape)
        torch.save(ade_fde, './curve/' + str(self.nm) + 'ft.pt')
        plt.plot(x, np.array(a)[:])
        plt.plot(x, np.array(f)[:])

        plt.savefig(str(self.nm)+'ft.png')
        plt.show()


        #print( 'll', loss_all /samples,samples)
        print(vb)
        print(beta ,samples, dict_metrics['ade_48s'],dict_metrics['fde_48s'],',' )

        return 0


