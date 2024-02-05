import os
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import LinearSegmentedColormap
import datetime
import cv2
from torch.optim.lr_scheduler import LambdaLR
from random import randint
import copy
import torch.nn.functional as F
import random
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
import dataset_invariance
import index_qualitative
import tqdm
from data.loader import data_loader
from utils import (
    displacement_error,
    final_displacement_error,
    mse_error,
    get_dset_path,
    int_tuple,
    l2_loss,
    relative_to_abs,
)
import logging
def cal_de_mse(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt, mode="raw")
    de = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode="raw")
    mse = mse_error(pred_traj_fake, pred_traj_gt)
    return ade, de, mse
def evaluate_helper(error, seq_start_end):
    sum_ = 0

    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_
class Trainer:
    def __init__(self, config):
        """
        Trainer class for training the Iterative Refinement Module (IRM)
        :param config: configuration parameters (see train_IRM.py)
        """

        self.index_qualitative = index_qualitative.dict_test
        self.name_run = 'runs/runs-IRM/'
        self.name_test = str(datetime.datetime.now())[:19]
        self.folder_test = 'training/training_IRM/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
        self.file = open(self.folder_test + "details.txt", "w")
        tracks = json.load(open(config.dataset_file))

        self.dim_clip = 180
        print('creating dataset...')
        train_path = get_dset_path(config.dataset_name, "train")  # train
        val_path = get_dset_path(config.dataset_name, "test")  ##test
        val_= get_dset_path(config.dataset_name, "val")

        logging.info("Initializing train dataset")
        train_dset, train_loader = data_loader(config, train_path)
        logging.info("Initializing test dataset")
        test_set, val_loader = data_loader(config, val_path)
        logging.info("Initializing val dataset")
        val_set, valoader = data_loader(config, val_)
        self.data_train=train_dset
        self.data_val=val_set
        self.data_test=test_set
        self.train_loader = train_loader
        self.val_loader = valoader
        self.mem_n2n_trained = torch.load(config.mantra_model)
        self.base_weights = self.mem_n2n_trained.parameters(
        )

        #self.mem_n2n = torch.load(config.model)


        self.test_loader = val_loader
        print('dataset created')

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
        self.mem_n2n_train = torch.load(config.mantra_model)
        # load model to evaluate
        self.mem_n2n_train.num_prediction = config.best_k
        self.mem_n2n_train.future_len = config.future_len
        self.mem_n2n_train.past_len = config.past_len


        if config.cuda:
            self.mem_n2n_train = self.mem_n2n_train.cuda()


        self.model = torch.load(config.model)
        self.mem_n2n = model_memory_IRM(self.settings, self.model)
        self.mem_n2n.past_len = config.past_len
        self.mem_n2n.future_len = config.future_len

        self.criterionLoss = nn.MSELoss()
        #self.opt = torch.optim.Adam(self.mem_n2n.parameters(), lr=config.learning_rate)
        self.iterations = 0
        if config.cuda:
            self.criterionLoss = self.criterionLoss.cuda()
            self.mem_n2n = self.mem_n2n.cuda()

        self.start_epoch = 0
        self.config = config
        self.pred_len=config.future_len
        # Write details to file
        self.write_details()
        self.file.close()
        self.best_ade=100
        # Tensorboard summary: configuration
        self.writer = SummaryWriter(self.name_run + self.name_test + '_' + config.info)
        self.writer.add_text('Training Configuration', 'model name: ' + self.mem_n2n.name_model, 0)
        self.writer.add_text('Training Configuration', 'dataset train: ' + str(len(self.data_train)), 0)
        self.writer.add_text('Training Configuration', 'dataset test: ' + str(len(self.data_test)), 0)
        self.writer.add_text('Training Configuration', 'number of prediction: ' + str(self.num_prediction), 0)
        self.writer.add_text('Training Configuration', 'batch_size: ' + str(self.config.batch_size), 0)
        self.writer.add_text('Training Configuration', 'learning rate init: ' + str(self.config.learning_rate), 0)
        self.writer.add_text('Training Configuration', 'dim_embedding_key: ' + str(self.settings["dim_embedding_key"]),
                             0)

    def write_details(self):
        """
        Serialize configuration parameters to file.
        """
        self.file.write("points of past track: " + str(self.config.past_len) + '\n')
        self.file.write("points of future track: " + str(self.config.future_len) + '\n')
        self.file.write("train size: " + str(len(self.data_train)) + '\n')
        self.file.write("test size: " + str(len(self.data_test)) + '\n')
        self.file.write("batch size: " + str(self.config.batch_size) + '\n')

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
            param.requires_grad =False

        for param in self.mem_n2n.encoder_fut.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.linear_controller.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.decoder.parameters():
            param.requires_grad = True
        for param in self.mem_n2n.FC_output.parameters():
            param.requires_grad = True


        for param in self.mem_n2n.Lin_Q.parameters():
            param.requires_grad = True
        for param in self.mem_n2n.Lin_K.parameters():
            param.requires_grad = True
        for param in self.mem_n2n.Lin_V.parameters():
            param.requires_grad = True


        # Load memory
        # populate the memory
        start = time.time()
        self._memory_writing(self.config.saved_memory)
        self.writer.add_text('Training Configuration', 'memory size: ' + str(len(self.mem_n2n.memory_past)), 0)
        end = time.time()
        print('writing time: ' + str(end-start))

        step_results = [1, 10, 20, 30, 40, 50, 60, 80, 90, 100, 120, 150, 170, 200, 250, 300, 350, 400, 450, 490, 550, 600]
        # Main training loop
        from torch.nn import init
        init.uniform_(self.mem_n2n.Lin_Sgmoid.weight, a=0.99, b=1.0)
        init.constant_(self.mem_n2n.Lin_Sgmoid.bias, val=0.0)
        for epoch in range(600):#self.start_epoch, config.max_epochs
            self.mem_n2n.train()

            print('epoch: ' + str(epoch))
            start = time.time()


            loss = self._train_single_epoch()

            end = time.time()
            print('Epoch took: {} Loss: {}'.format(end - start, loss))

            #if (epoch + 1) in step_results:
            # Test model while training
            print('start test')
            start_test = time.time()
            dict_metrics_test = self.evaluate(self.test_loader, epoch + 1)
            end_test = time.time()
            print('Test took: {}'.format(end_test - start_test))
            print(dict_metrics_test)

            # Tensorboard summary: test
            #self.writer.add_scalar('accuracy_test/euclMean', dict_metrics_test['euclMean'], epoch)
            #self.writer.add_scalar('accuracy_test/Horizon40s', dict_metrics_test['horizon40s'], epoch)
            #self.writer.add_scalar('dimension_memory/memory', len(self.mem_n2n.memory_past), epoch)

            # Save model checkpoint
            '''if dict_metrics_test['euclMean'].item()<self.best_ade:
                print('ade:',dict_metrics_test['euclMean'],'fde:',dict_metrics_test['horizon40s'])
                self.best_ade=dict_metrics_test['euclMean'].item()
                torch.save(self.mem_n2n, self.folder_test + 'model_IRM_epoch_' + str(epoch) + '_' + self.name_test)
                self.save_results(dict_metrics_test, epoch=epoch + 1)
            else:print('ade:', dict_metrics_test['euclMean'], 'fde:', dict_metrics_test['horizon40s'])
            for name, param in self.mem_n2n.named_parameters():
                self.writer.add_histogram(name, param.data, epoch)'''

        # Save final trained model
        torch.save(self.mem_n2n, self.folder_test + 'model_mantra_' + self.name_test)

    def save_results(self, dict_metrics_test, epoch=0):
        """
        Serialize results
        :param dict_metrics_test: dictionary with test metrics
        :param epoch: epoch index (default: 0)
        :return: None
        """
        self.file = open(self.folder_test + "results.txt", "w")
        self.file.write("TEST:" + '\n')
        #self.file.write("split test: " + str(self.data_test.ids_split_test) + '\n')
        self.file.write("num_predictions:" + str(self.config.preds) + '\n')
        self.file.write("memory size: " + str(len(self.mem_n2n.memory_past)) + '\n')
        self.file.write("epoch: " + str(epoch) + '\n')


        self.file.write("error 4s: " + str(dict_metrics_test['horizon40s'].item()) + '\n')

        self.file.write("ADE 4s: " + str(dict_metrics_test['euclMean'].item()) + '\n')

        self.file.close()

    def replay_init(self, data_train,past_emb):
        """
        Initialization: write samples in memory.
        :param data_train: dataset
        :return: None
        """

        past = torch.FloatTensor()
        past_rel = torch.FloatTensor()
        fut = torch.FloatTensor()
        fut_rel = torch.FloatTensor()
        non_linear_ped = torch.FloatTensor()
        loss_mask = torch.FloatTensor()
        ff = []

        for _ in range(300):
            j = random.randint(0, len(data_train) - 1)
            past = torch.cat((past, data_train[j][0]), dim=0)
            fut = torch.cat((fut, data_train[j][1]), dim=0)
            past_rel = torch.cat((past_rel, data_train[j][2]), dim=0)
            fut_rel = torch.cat((fut_rel, data_train[j][3]), dim=0)
            ff.append(data_train[j][1])
            non_linear_ped = torch.cat((non_linear_ped, data_train[j][4]), dim=0)
            loss_mask = torch.cat((loss_mask, data_train[j][5]), dim=0)
        _len = [len(seq) for seq in ff]
        cum_start_idx = [0] + np.cumsum(_len).tolist()
        seq_start_end = [
            [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]


        key_sort = F.normalize(past_rel.reshape(-1, 16), p=2, dim=1)
        query_sort = F.normalize(past_emb.reshape(-1, 16), p=2, dim=1)
        score = torch.matmul(query_sort, key_sort.t())  # (bs,m)
        _, index = torch.topk(score, 1, dim=1)

        past = past[index].squeeze().reshape(-1, 8, 2)
        future = fut[index].squeeze().reshape(-1, 12, 2)
        past_rel = past_rel[index].squeeze().reshape(-1, 8, 2)
        future_rel = fut_rel[index].squeeze().reshape(-1, 12, 2)

        # return past[sample_random].squeeze(),future[sample_random].squeeze(), past_rel[sample_random].squeeze(),future_rel[sample_random].squeeze()

        return past, future, past_rel, future_rel

    def replay_memory(self,past_emb,old_data_len):
         future= self.mem_n2n_train.pred_gt[old_data_len-21:]
         past = self.mem_n2n_train.past_gt[old_data_len-21:]
         future_rel = self.mem_n2n_train.pred_rel_gt[old_data_len-21:]
         past_rel = self.mem_n2n_train.past_rel_gt[old_data_len-21:]

         past_emb = torch.transpose(past_emb, 1, 2)
         story_embed = self.mem_n2n_train.relu(self.mem_n2n_train.conv_past(past_emb))
         story_embed = torch.transpose(story_embed, 1, 2)
         output_past, state_past = self.mem_n2n_train.encoder_past(story_embed)

         # past temporal encoding
         past_emb_old = torch.transpose(past_rel, 1, 2)
         story_emb_old = self.mem_n2n_train.relu(self.mem_n2n_train.conv_past(past_emb_old))
         story_emb_old = torch.transpose(story_emb_old, 1, 2)
         output_past, repaly_past = self.mem_n2n_train.encoder_past(story_emb_old)


         key_sort = F.normalize(repaly_past.squeeze(), p=2, dim=1)
         query_sort = F.normalize(state_past.squeeze(), p=2, dim=1)

         score = torch.matmul(query_sort, key_sort.t())  # (bs,m)


         ff, index = torch.topk(score,20, dim=1) #(bs,10)

         index=torch.unique(index.reshape(-1),dim=0).reshape(-1,1)

         past=past[index].squeeze()#.reshape(-1,8,2)

         ### delete repeat
         future = future[index].squeeze().reshape(-1,12,2)

         past_rel = past_rel[index].squeeze().reshape(-1,8,2)
         future_rel = future_rel[index].squeeze().reshape(-1,12,2)

        # return past[sample_random].squeeze(),future[sample_random].squeeze(), past_rel[sample_random].squeeze(),future_rel[sample_random].squeeze()

         return past, future, past_rel, future_rel

    def evaluate(self, loader, epoch=0):
        """
        Evaluate model. Future trajectories are predicted and
        :param loader: data loader for testing data
        :param epoch: epoch index (default: 0)
        :return: dictionary of performance metrics
        """

        self.mem_n2n.eval()
        data_len=0
        with torch.no_grad():
            dict_metrics = {}
            ade = utils.AverageMeter("ADE", ":.6f")
            fde = utils.AverageMeter("FDE", ":.6f")
            ade_outer, de_outer, mse_outer = [], [], []
            progress = utils.ProgressMeter(len(loader), [ade, fde], prefix="Test: ")

            total_traj = 0
            eucl_mean = ADE_1s = ADE_2s = ADE_3s = horizon10s = horizon20s = horizon30s = horizon40s = 0

            for step, batch in enumerate(tqdm.tqdm(loader)):

                batch = [tensor.cuda() for tensor in batch]
                (
                    obs_traj,
                    pred_traj_gt,
                    obs_traj_rel,
                    pred_traj_gt_rel,
                    non_linear_ped,
                    loss_mask,
                    seq_start_end,
                ) = batch

                past = Variable(obs_traj)
                past = past.transpose(1, 0)
                future = Variable(pred_traj_gt)
                future = future.transpose(1, 0)

                past_rel = Variable(obs_traj_rel)
                past_rel = past_rel.transpose(1, 0)
                future_rel = Variable(pred_traj_gt_rel)
                future_rel = future_rel.transpose(1, 0)



                if self.config.cuda:
                    past = past.cuda()
                    future = future.cuda()
                    past_rel = past_rel.cuda()
                    future_rel = future_rel.cuda


                pred = self.mem_n2n(past_rel,obs_traj=past)
                ade1, de1, mse1 = [], [], []

                total_traj += pred_traj_gt.size(1)

                for topki in range(self.config.best_k):  # topk=20( num_samples is topk )
                    # multi-modal
                    pred_traj_fake = pred[:, topki]  # ([729, 20, 12, 2])
                    # pred_traj_fake_rel = pred_traj_fake_rel[-self.pred_len:]
                    pred_traj_fake = pred_traj_fake.transpose(1,0)  # relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                    ade_, de_, mse_ = cal_de_mse(pred_traj_gt, pred_traj_fake)
                    ade1.append(ade_)
                    de1.append(de_)
                    mse1.append(mse_)
                # print('ff',mem.size())
                ade_sum = evaluate_helper(ade1, seq_start_end)
                de_sum = evaluate_helper(de1, seq_start_end)
                mse_sum = evaluate_helper(mse1, seq_start_end)

                ade_outer.append(ade_sum)
                de_outer.append(de_sum)
                mse_outer.append(mse_sum)

            ade1 = sum(ade_outer) / (total_traj * self.pred_len)
            de1 = sum(de_outer) / (total_traj)
            mse1 = sum(mse_outer) / (total_traj)

            ade.update(ade1, obs_traj.shape[1])
            fde.update(de1, obs_traj.shape[1])


            dict_metrics['euclMean'] = torch.tensor(ade.avg,dtype=torch.float)
            dict_metrics['horizon40s'] = torch.tensor(fde.avg,dtype=torch.float)

        return dict_metrics

    def _train_single_epoch(self):
        """
        Training loop over the dataset for an epoch
        :return: loss
        """
        config = self.config


        ade = utils.AverageMeter("ADE", ":.6f")
        fde = utils.AverageMeter("FDE", ":.6f")
        ade_outer, de_outer, mse_outer = [], [], []
        progress = utils.ProgressMeter(len(self.test_loader), [ade, fde], prefix="Test: ")

        total_traj = 0

        base_weights = torch.load(config.mantra_model)
        base_weights = list(base_weights.parameters(
        ))


        #torch.backends.cudnn.enabled = False
        base_weights = torch.load(config.mantra_model)
        base_weights = list(base_weights.parameters(
        ))

        old_data_len = len(self.mem_n2n_train.past_gt)
        for step, batch in enumerate(tqdm.tqdm(self.test_loader)):


            self.iterations += 1

            batch = [tensor.cuda() for tensor in batch]
            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_gt_rel,
                non_linear_ped,
                loss_mask,
                seq_start_end,
            ) = batch

            past  = Variable(obs_traj)
            past  = past.transpose(1, 0)
            future = Variable(pred_traj_gt)
            future = future.transpose(1, 0)

            past_rel = Variable(obs_traj_rel)
            past_rel = past_rel.transpose(1, 0)
            future_rel = Variable(pred_traj_gt_rel)
            future_rel = future_rel.transpose(1, 0)


            mem_n2n = copy.deepcopy(self.mem_n2n_train)
            mem_n2n.train()
            opt = torch.optim.Adam(mem_n2n.parameters(), lr=5e-4)




            if  self.config.cuda:
                past = past.cuda()
                future = future.cuda()
                past_rel = past_rel.cuda()
                future_rel = future_rel.cuda()
            # old data

            l2_loss_rel_train = []
            past_train, future_train, past_rel_train, future_rel_train = self.replay_memory(past_rel,old_data_len)  # self.replay_init(self.data_train)

            past_train = Variable(past_train).cuda()

            future_train = Variable(future_train).cuda()

            past_rel_train = Variable(past_rel_train).cuda()

            future_rel_train = Variable(future_rel_train).cuda()
            curr_weights = list(mem_n2n.parameters())

            for ii in range(60):

                opt.zero_grad()

                # Current model weights
                loss = torch.zeros(1).cuda()
                diff_loss = torch.Tensor([0]).cuda()
                # Iterate over base_weights and curr_weights and accumulate the euclidean norm
                # of their differences

                for base_param, curr_param in zip(base_weights, curr_weights):
                    diff_loss += (curr_param - base_param).pow(2).sum()


                output_train = mem_n2n(past_rel_train, obs_traj=past_train)
                # args.pred_len, mem, epoch,
                l2_loss_sum_rel_train = torch.zeros(1).cuda()
                rmse_all = []
                for topki in range(config.best_k):
                    pred_traj_fake_train = output_train[:, topki]  # ([729, 20, 12, 2])
                    # pred_traj_fake_rel = pred_traj_fake_rel[-self.pred_len:]

                    # rmse
                    batch, seq_len, _ = pred_traj_fake_train.size()
                    # equation below , the first part do noing, can be delete

                    loss_f = (pred_traj_fake_train - future_train) ** 2

                    rmse = (loss_f.sum(dim=2).sum(dim=1) / seq_len).reshape(-1, 1)

                    rmse_all.append(rmse)

                rmse_ = torch.stack(rmse_all, dim=1).squeeze()

                best, _ = torch.min(rmse_, dim=1)

                l2_loss_sum_rel_train = best.sum() #torch.mean(best).reshape(1).cuda()


                loss = loss    + l2_loss_sum_rel_train #+ 1*diff_loss

                loss.backward()
                #torch.nn.utils.clip_grad_norm_(mem_n2n.parameters(), 1.0, norm_type=2)
                opt.step()

            print(loss)
            self.writer.add_scalar('loss/loss_total', loss, self.iterations)


            ade1, de1, mse1 = [], [], []
            mem_n2n.eval()
            pred = mem_n2n(past_rel, obs_traj=past)

            total_traj += pred_traj_gt.size(1)

            for topki in range(self.config.best_k):  # topk=20( num_samples is topk )
                # multi-modal
                pred_traj_fake = pred[:, topki]  # ([729, 20, 12, 2])
                # pred_traj_fake_rel = pred_traj_fake_rel[-self.pred_len:]
                pred_traj_fake = pred_traj_fake.transpose(1, 0)  # relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                ade_, de_, mse_ = cal_de_mse(pred_traj_gt, pred_traj_fake)
                ade1.append(ade_)
                de1.append(de_)
                mse1.append(mse_)

            ade_sum = evaluate_helper(ade1, seq_start_end)
            de_sum = evaluate_helper(de1, seq_start_end)
            mse_sum = evaluate_helper(mse1, seq_start_end)
            print('a',ade_sum/(pred_traj_gt.size(1)*12.0))
            # write current samples

            self.mem_n2n_train.write_in_memory(past_rel, future=future_rel,obs_traj=past,pred_gt=future)


            print(self.mem_n2n_train.memory_past.shape,self.mem_n2n_train.past_gt.shape)


            ade_outer.append(ade_sum)
            de_outer.append(de_sum)
            mse_outer.append(mse_sum)



        ade1 = sum(ade_outer) / (total_traj * self.pred_len)
        de1 = sum(de_outer) / (total_traj)
        mse1 = sum(mse_outer) / (total_traj)
        ade.update(ade1, obs_traj.shape[1])
        fde.update(de1, obs_traj.shape[1])
        print('ade:',ade.avg,' fde:',fde.avg)



        return loss.item()

    def _memory_writing(self, saved_memory):
        """
        writing in the memory with controller (loop over all train dataset)
        :return: loss
        """

        if saved_memory:
            print('memories of pretrained model')
            #self.mem_n2n.memory_past = torch.load('./pretrained_models/MANTRA/increm_mem/memory_past.pt')
            #self.mem_n2n.memory_fut = torch.load('./pretrained_models/MANTRA/increm_mem/memory_fut.pt')
            print('a', self.mem_n2n.memory_past.shape,self.mem_n2n.pred_rel_gt.shape)

        else:
            self.mem_n2n.init_memory(self.data_train)
            #config = self.config
            with torch.no_grad():
                #### train set

                for step, batch in enumerate(tqdm.tqdm(self.train_loader)):
                    self.iterations += 1
                    batch = [tensor.cuda() for tensor in batch]
                    (
                        obs_traj,
                        pred_traj_gt,
                        obs_traj_rel,
                        pred_traj_gt_rel,
                        non_linear_ped,
                        loss_mask,
                        seq_start_end,
                    ) = batch

                    past = Variable(obs_traj)
                    past = past.transpose(1, 0)
                    future = Variable(pred_traj_gt)
                    future = future.transpose(1, 0)

                    past_rel = Variable(obs_traj_rel)
                    past_rel = past_rel.transpose(1, 0)
                    future_rel = Variable(pred_traj_gt_rel)
                    future_rel = future_rel.transpose(1, 0)

                    if self.config.cuda:
                        past = past.cuda()
                        future_ = future.cuda()
                        past_rel = past_rel.cuda()
                        future_rel = future_rel.cuda()

                    self.mem_n2n.write_in_memory(past_rel, future=future_rel,obs_traj=past,pred_gt=future_)
            print('a', self.mem_n2n.memory_past.shape)
            with torch.no_grad():
                ### val set
                for step, batch in enumerate(tqdm.tqdm(self.val_loader)):
                    self.iterations += 1

                    batch = [tensor.cuda() for tensor in batch]
                    (
                        obs_traj,
                        pred_traj_gt,
                        obs_traj_rel,
                        pred_traj_gt_rel,
                        non_linear_ped,
                        loss_mask,
                        seq_start_end,
                    ) = batch

                    past = Variable(obs_traj)
                    past = past.transpose(1, 0)
                    future = Variable(pred_traj_gt)
                    future = future.transpose(1, 0)

                    past_rel = Variable(obs_traj_rel)
                    past_rel = past_rel.transpose(1, 0)
                    future_rel = Variable(pred_traj_gt_rel)
                    future_rel = future_rel.transpose(1, 0)

                    if self.config.cuda:
                        past = past.cuda()
                        future = future.cuda()
                        past_rel = past_rel.cuda()
                        future_rel = future_rel.cuda()

                    self.mem_n2n.write_in_memory(past_rel, future_rel,obs_traj=past,pred_gt=future)
        print('n', self.mem_n2n.memory_past.shape)
        # save memory
        torch.save(self.mem_n2n.memory_past, self.folder_test + 'memory_past.pt')
        torch.save(self.mem_n2n.memory_fut, self.folder_test + 'memory_fut.pt')

