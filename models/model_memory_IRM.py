import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import math

class Encoder_F(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""

    def __init__(
            self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
            dropout=0.0
    ):
        super(Encoder_F, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.GRU(
            embedding_dim, h_dim, num_layers, dropout=dropout  # embedding_dim indicates the dimension of the data input into the LSTM
        )

        self.spatial_embedding = nn.Linear(2, embedding_dim)

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.Lin=nn.Linear(48,48)

    def init_hidden(self, batch):  # initialize h0 and c0
        return torch.zeros(self.num_layers, batch, self.h_dim).cuda()
            #torch.zeros(self.num_layers, batch, self.h_dim).cuda()


    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch, self.embedding_dim
        )
        state_tuple = self.init_hidden(batch)
        # input of LSTM: input_data (seq_len, batch, input_size), h0, and c0
        # output of LSTM: output(seq_len, batch, hidden_size * num_directions), hn, and cn
        output, state = self.encoder(self.leaky_relu(obs_traj_embedding), state_tuple)#[8, 852, 48]) torch.Size([1, 852, 48]
        #final_h = state # use the hidden states (hn) of the LSTM
        ######### ???
        final_h = state#.permute(1, 2, 0)
        #final_h = self.Lin(final_h)


        return final_h#.view(1,-1,96)#(12,batch,48)


class model_memory_IRM(nn.Module):
    """ade: tensor(0.3849) fde: tensor(0.7905)
    Memory Network model with Iterative Refinement Module.
    """

    def __init__(self, settings, model_pretrained):
        super(model_memory_IRM, self).__init__()
        self.name_model = 'MANTRA'

        # parameters
        self.use_cuda = settings["use_cuda"]
        self.dim_embedding_key = settings["dim_embedding_key"]
        self.num_prediction = settings["num_prediction"]
        self.past_len = settings["past_len"]
        self.future_len = settings["future_len"]
        self.th = settings["th"]
        # similarity criterion
        self.weight_read = []
        self.index_max = []
        self.similarity = nn.CosineSimilarity(dim=1)

        # Memory
        self.memory_past = model_pretrained.memory_past
        self.memory_fut  =  model_pretrained.memory_fut

        '''self.pred_gt  = model_pretrained.pred_gt
        self.past_gt  = model_pretrained.past_gt
        self.pred_rel_gt = model_pretrained.pred_rel_gt
        self.past_rel_gt = model_pretrained.past_rel_gt'''


        self.memory_count = []

        # layers
        self.conv_past = model_pretrained.conv_past
        self.conv_fut = model_pretrained.conv_fut

        self.encoder_past = model_pretrained.encoder_past
        self.encoder_fut = model_pretrained.encoder_fut
        self.decoder = model_pretrained.decoder
        self.FC_output = model_pretrained.FC_output

        self.decoder_fut = nn.GRU(self.dim_embedding_key * 2, self.dim_embedding_key * 2, num_layers=1)#model_pretrained.decoder
        self.FC_output_fut = torch.nn.Linear(self.dim_embedding_key * 2, 2)#model_pretrained.FC_output
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.maxpool2d = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # writing controller
        self.linear_controller = model_pretrained.linear_controller

        # scene: input shape (batch, classes, 360, 360)
        self.convScene_1 = nn.Sequential(nn.Conv2d(4, 8, kernel_size=5, stride=2, padding=2), nn.ReLU(),
                                         nn.BatchNorm2d(8))
        self.convScene_2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(), nn.BatchNorm2d(16))

        self.RNN_scene = nn.GRU(16, self.dim_embedding_key, 1, batch_first=True)

        # refinement fc layer
        self.fc_refine = nn.Linear(self.dim_embedding_key, self.future_len * 2)

        self.reset_parameters()
        self.Lin_Q = nn.Linear(48,48)
        self.Lin_K = nn.Linear(48,48)
        self.Lin_V = nn.Linear(48,48)
        self.Lin_Sgmoid=nn.Linear(48,1)

        '''
        self.Lin_Q = Encoder_F(
            embedding_dim=48,
            h_dim=48,#48
            mlp_dim=1024,
            num_layers=1,
            dropout=0
        )
        '''
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.RNN_scene.weight_ih_l0)
        nn.init.kaiming_normal_(self.RNN_scene.weight_hh_l0)
        nn.init.kaiming_normal_(self.RNN_scene.weight_ih_l0)
        nn.init.kaiming_normal_(self.RNN_scene.weight_hh_l0)
        nn.init.kaiming_normal_(self.convScene_1[0].weight)
        nn.init.kaiming_normal_(self.convScene_2[0].weight)
        nn.init.kaiming_normal_(self.fc_refine.weight)

        nn.init.zeros_(self.RNN_scene.bias_ih_l0)
        nn.init.zeros_(self.RNN_scene.bias_hh_l0)
        nn.init.zeros_(self.RNN_scene.bias_ih_l0)
        nn.init.zeros_(self.RNN_scene.bias_hh_l0)
        nn.init.zeros_(self.convScene_1[0].bias)
        nn.init.zeros_(self.convScene_2[0].bias)
        nn.init.zeros_(self.fc_refine.bias)

    def init_memory(self, data_train):
        """
        Initialization: write element in memory.
        :param data_train: dataset
        :return: None
        """

        # random element from train dataset to be added in memory
        j = random.randint(0, len(data_train) - 1)
        past = data_train[j][0]
        future = data_train[j][1]

        k = random.randint(0, len(data_train[j][1]) - 1)
        past = past[k].transpose(0, 1)
        past = past.cuda().unsqueeze(0)
        future = future[k].transpose(0, 1)
        future = future.cuda().unsqueeze(0)

        # past encoding
        past = torch.transpose(past, 1, 2)
        story_embed = self.relu(self.conv_past(past))
        story_embed = torch.transpose(story_embed, 1, 2)
        output_past, state_past = self.encoder_past(story_embed)

        # future encoding
        future = torch.transpose(future, 1, 2)
        future_embed = self.relu(self.conv_fut(future))
        future_embed = torch.transpose(future_embed, 1, 2)
        output_fut, state_fut = self.encoder_fut(future_embed)

        # insert in memory
        self.memory_past = torch.cat((self.memory_past, state_past.squeeze(0)), 0)
        self.memory_fut = torch.cat((self.memory_fut, state_fut.squeeze(0)), 0)

    # #ablation study
        # future = torch.transpose(future, 1, 2)
        # self.memory_count = torch.cat((self.memory_count, future), 0)

    def forward(self, past, scene=None,obs_traj=None,past_embed=None):
        """
        Forward pass. Refine predictions generated by MemNet with IRM.
        :param past: past trajectory
        :param scene: surrounding map
        :return: predicted future
        """
        self.encoder_past.flatten_parameters()

        self.decoder_fut.flatten_parameters()
        past_org = past
        dim_batch = past.size()[0]
        zero_padding = torch.zeros(1, dim_batch * self.num_prediction, self.dim_embedding_key * 2).cuda()
        prediction = torch.Tensor().cuda()
        present_temp = obs_traj[:, -1].unsqueeze(1)  # past[:, -1].unsqueeze(1)

        # past temporal encoding
        past = torch.transpose(past, 1, 2)
        story_embed = self.relu(self.conv_past(past))
        story_embed = torch.transpose(story_embed, 1, 2)
        output_past, state_past = self.encoder_past(story_embed)

        # Cosine similarity and memory read
        pred = []
        pred_val = []
        all = state_past.view(-1, 1, 48)

        ###
        if past_embed is not None:
            state_past = past_embed.view(1, -1, 48)
            all = state_past.view(-1, 1, 48)
        c = all
        memory_past = self.memory_past
        memory_fut = self.memory_fut
        mem_batch = memory_fut.size(0)


        key_topk = memory_past  # [index[:, :n]]
        value_topk = memory_fut  # [index[:, :n]]

        for _ in range(20):
            # [:, :n]n (bs,n,48)

            key_random = self.Lin_K(key_topk)
            value_random = self.Lin_V(value_topk)
            ### rand_select from memory
            # select topk
            all = c

            query = self.Lin_Q(all)
            query = query.reshape(-1, 48)

            # prob attention
            dim = 48
            scale = 1.0  # / math.sqrt(dim)

            v = torch.matmul(query, key_random.transpose(0, 1))

            v_scal = scale * v  # (bs,1,10)
            v = torch.softmax(v_scal, dim=-1)

            future_rel_ = torch.matmul(v, value_random)

            att = future_rel_.reshape(-1, 48)  # batch, n, dim_v

            query = query.reshape(-1, 48) + att.reshape(-1, 48)
            c = query

            pred.append(att)
        pred = torch.stack(pred, dim=0)
        ###############################################

        pred = pred.transpose(1, 0)
        present = present_temp.repeat_interleave(self.num_prediction, dim=0)

        state_past = state_past.repeat_interleave(self.num_prediction, dim=1)
        info_future = pred.reshape(1, -1, 48)

        info_total = torch.cat((state_past, info_future), 2)
        input_dec = info_total
        state_dec = zero_padding

        for i in range(self.future_len):
            output_decoder, state_dec = self.decoder_fut(input_dec, state_dec)

            displacement_next = self.FC_output_fut(output_decoder)
            coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
            prediction = torch.cat((prediction, coords_next), 1)
            present = coords_next
            input_dec = zero_padding

        prediction = prediction.view(dim_batch, self.num_prediction, self.future_len, 2)

        return prediction

    def write_in_memory(self, past, future, obs_traj=None, pred_gt=None):
        """
               Writing controller decides if the pair past-future will be inserted in memory.
               :param past: past trajectory
               :param future: future trajectory
               """
        past_org = past
        future_org = future

        if self.memory_past.shape[0] < self.num_prediction:
            num_prediction = self.memory_past.shape[0]
        else:
            num_prediction = self.num_prediction


        dim_batch = past.size()[0]
        zero_padding = torch.zeros(1, dim_batch, self.dim_embedding_key * 2).cuda()
        prediction = torch.Tensor().cuda()
        present_temp = obs_traj[:, -1].unsqueeze(1)  # past[:, -1].unsqueeze(1)

        # past temporal encoding
        past = torch.transpose(past, 1, 2)
        story_embed = self.relu(self.conv_past(past))
        story_embed = torch.transpose(story_embed, 1, 2)
        output_past, state_past = self.encoder_past(story_embed)

        # Cosine similarity and memory read
        past_normalized = F.normalize(self.memory_past, p=2, dim=1)
        state_normalized = F.normalize(state_past.reshape(-1,48), p=2, dim=1)
        weight_read = torch.matmul(past_normalized, state_normalized.transpose(0, 1)).transpose(0, 1)
        index_max = torch.sort(weight_read, descending=True)[1].cpu()[:, :num_prediction]

        for i_track in range(num_prediction):
            present = present_temp
            prediction_single = torch.Tensor().cuda()
            ind = index_max[:, i_track]
            info_future = self.memory_fut[ind]
            info_total = torch.cat((state_past, info_future.unsqueeze(0)), 2)
            input_dec = info_total
            state_dec = zero_padding
            for i in range(self.future_len):
                output_decoder, state_dec = self.decoder(input_dec, state_dec)
                displacement_next = self.FC_output(output_decoder)
                coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
                prediction_single = torch.cat((prediction_single, coords_next), 1)
                present = coords_next
                input_dec = zero_padding
            prediction = torch.cat((prediction, prediction_single.unsqueeze(1)), 1)

        future_rep = pred_gt.unsqueeze(1).repeat(1, num_prediction, 1, 1)

        distances = torch.norm(prediction - future_rep, dim=3)

        th = self.th

        tolerance_1s = torch.sum(distances[:, :, :3] < th * 0.25, dim=2)
        tolerance_2s = torch.sum(distances[:, :, 3:6] < th * 0.5, dim=2)
        tolerance_3s = torch.sum(distances[:, :, 6:9] < th * 0.75, dim=2)
        tolerance_4s = torch.sum(distances[:, :, 9:12] < th, dim=2)
        tolerance = tolerance_1s + tolerance_2s + tolerance_3s + tolerance_4s

        #fde = torch.sum(distances[:, :, -1].view(-1, 20, 1) >= 1.3 * th, dim=-1)
        #fde = fde.type(torch.FloatTensor)
        tolerance = (tolerance_1s + tolerance_2s + tolerance_3s + tolerance_4s)
        tolerance = tolerance.type(torch.FloatTensor) / 12.0

        tolerance = 1.0 - tolerance.type(torch.FloatTensor)
        #tolerance = tolerance + fde
        #tolerance = tolerance / 2.0

        tolerance_rate, inde = torch.max(tolerance.type(torch.FloatTensor), dim=1)
        # toler = tolerance_rate.cuda() + fde[inde].cuda()
        batch = inde.shape[0]

        # FDE
        #dist = tolerance_rate + fde[torch.arange(batch)[:, None], inde.view(-1, 1)].squeeze()
        #dist = dist / 2.0

        # print(distances[:, :, :8] < 100*0.25,tolerance_rate)torch.Size([25294, 48])
        # final destination

        tolerance_rate = tolerance_rate.unsqueeze(1).cuda()  # dist.unsqueeze(1).cuda()
        # controller
        writing_prob = torch.sigmoid(self.linear_controller(tolerance_rate))

        # future encoding
        future = torch.transpose(future, 1, 2)
        future_embed = self.relu(self.conv_fut(future))
        future_embed = torch.transpose(future_embed, 1, 2)
        output_fut, state_fut = self.encoder_fut(future_embed)

        # index of elements to be added in memory
        index_writing = np.where(writing_prob.cpu() > 0.65)[0]

        batch_size = past_org.shape[0]
        index_writing2 = np.random.randint(batch_size, size=int(batch_size / 20.0))


        past_to_write = state_past.reshape(-1,48)[index_writing]
        future_to_write = state_fut.reshape(-1,48)[index_writing]

        pred_rel_gt_write = future_org[index_writing2].reshape(-1, 12, 2)
        past_rel_gt_write = past_org[index_writing2].reshape(-1, 8, 2)

        pred_gt_write = pred_gt[index_writing2].reshape(-1, 12, 2)
        past_gt_write = obs_traj[index_writing2].reshape(-1, 8, 2)

        self.memory_past = torch.cat((self.memory_past, past_to_write), 0)
        self.memory_fut = torch.cat((self.memory_fut, future_to_write), 0)

        self.pred_gt = torch.cat((self.pred_gt, pred_gt_write), 0)
        self.past_gt = torch.cat((self.past_gt, past_gt_write), 0)

        self.pred_rel_gt = torch.cat((self.pred_rel_gt, pred_rel_gt_write), 0)
        self.past_rel_gt = torch.cat((self.past_rel_gt, past_rel_gt_write), 0)

        return torch.cat((past_to_write, future_to_write), dim=1)

