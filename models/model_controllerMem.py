import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class model_controllerMem(nn.Module):
    """
    Memory Network model with learnable writing controller.
    """

    def __init__(self, settings, model_pretrained):
        super(model_controllerMem, self).__init__()
        self.name_model = 'writing_controller'

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
        self.memory_past = torch.Tensor().cuda()
        self.memory_fut = torch.Tensor().cuda()
        self.pred_gt = torch.zeros((self.num_prediction+1,12,2)).cuda()
        self.past_gt = torch.zeros((self.num_prediction + 1, 8, 2)).cuda()
        self.pred_rel_gt = torch.zeros((self.num_prediction + 1, 12, 2)).cuda()
        self.past_rel_gt = torch.zeros((self.num_prediction + 1, 8, 2)).cuda()


        self.memory_count = []

        # layers
        self.conv_past = model_pretrained.conv_past
        self.conv_fut = model_pretrained.conv_fut

        self.encoder_past = model_pretrained.encoder_past
        self.encoder_fut = model_pretrained.encoder_fut
        self.decoder = model_pretrained.decoder
        self.FC_output = model_pretrained.FC_output

        # activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.linear_controller = torch.nn.Linear(1, 1)


    def init_memory(self, data_train):
        """
        Initialization: write samples in memory.
        :param data_train: dataset
        :return: None
        """

        self.memory_past = torch.randn(21,48).cuda()
        self.memory_fut = torch.Tensor(21,48).cuda()


        '''for i in range(self.num_prediction + 1):

            # random element from train dataset to be added in memory
            j = random.randint(0, len(data_train)-1)
            past = data_train[j][2]
            future = data_train[j][3]


            k = random.randint(0, len(data_train[j][1]) - 1)
            past = past[k].transpose(0,1)
            past = past.cuda().unsqueeze(0)
            future = future[k].transpose(0,1)
            future = future.cuda().unsqueeze(0)
            ####


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
            self.memory_fut = torch.cat((self.memory_fut, state_fut.squeeze(0)), 0)'''

    def check_memory(self, index):
        """
        Method to generate a future track from past-future feature read from an index location of the memory.
        :param index: index of the memory
        :return: predicted future
        """

        mem_past_i = self.memory_past[index]
        mem_fut_i = self.memory_fut[index]
        zero_padding = torch.zeros(1, 1, 96).cuda()
        present = torch.zeros(1, 2).cuda()
        prediction_single = torch.Tensor().cuda()
        info_total = torch.cat((mem_past_i, mem_fut_i), 0)
        input_dec = info_total.unsqueeze(0).unsqueeze(0)
        state_dec = zero_padding
        for i in range(self.future_len):
            output_decoder, state_dec = self.decoder(input_dec, state_dec)
            displacement_next = self.FC_output(output_decoder)
            coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
            prediction_single = torch.cat((prediction_single, coords_next), 1)
            present = coords_next
            input_dec = zero_padding

        return prediction_single

    def forward(self, past,future=None ,obs_traj=None,pred_gt=None,scale=1):
        """
        Forward pass.
        Train phase: training writing controller based on reconstruction error of the future.
        Test phase: Predicts future trajectory based on past trajectory and the future feature read from the memory.
        :param past: past trajectory
        :param future: future trajectory (in test phase)
        :return: predicted future (test phase), writing probability and tolerance rate (train phase)
        """
        past_org=past
        future_org=future

        dim_batch = past.size()[0]
        zero_padding = torch.zeros(1, dim_batch, self.dim_embedding_key * 2)
        prediction = torch.Tensor()
        present_temp = obs_traj[:, -1].unsqueeze(1) #past[:, -1].unsqueeze(1)
        if self.use_cuda:
            zero_padding = zero_padding.cuda()
            prediction = prediction.cuda() # final

        # past temporal encoding
        past = torch.transpose(past, 1, 2)#(32,2,20)

        story_embed = self.relu(self.conv_past(past))
        story_embed = torch.transpose(story_embed, 1, 2)#(32,20,16)

        output_past, state_past = self.encoder_past(story_embed)
        s=state_past.shape[1]
        #print('f',output_past.size())f torch.Size([32, 20, 48])
        # Cosine similarity and memory read
        past_normalized = F.normalize(self.memory_past, p=2, dim=1)

        if s>1:
            state_normalized = F.normalize(state_past.squeeze(), p=2, dim=1)
        else:
            state_normalized = state_past.reshape(1,48)
        #state_normalized torch.Size([32, 48])
        self.weight_read = torch.matmul(past_normalized, state_normalized.transpose(0, 1)).transpose(0, 1)

        self.index_max = torch.sort(self.weight_read, descending=True)[1].cpu()

        for i_track in range(self.num_prediction):
            present = present_temp
            prediction_single = torch.Tensor().cuda()
            ind = self.index_max[:, i_track]
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


        if pred_gt is not None :#is not None


            prediction = prediction*scale

            future_rep =pred_gt.unsqueeze(1).repeat(1, self.num_prediction, 1, 1)
            distances = torch.norm(prediction - future_rep,p=2,dim=3)

            th=self.th
            tolerance_1s = torch.sum(distances[:, :, :3] < th*0.25, dim=2)
            tolerance_2s = torch.sum(distances[:, :, 3:6] < th*0.5, dim=2)
            tolerance_3s = torch.sum(distances[:, :, 6:9] < th*0.75, dim=2)
            tolerance_4s = torch.sum(distances[:, :, 9:12] < th, dim=2)
            fde = torch.sum( distances[:, :, -1].view(-1,20,1) >= 1.3*th , dim=-1 )
            fde=fde.type(torch.FloatTensor)
            tolerance = ( tolerance_1s + tolerance_2s + tolerance_3s + tolerance_4s )
            tolerance = tolerance.type(torch.FloatTensor) / 12.0

            tolerance = 1.0 - tolerance.type(torch.FloatTensor)

            tolerance_rate ,inde= torch.max(tolerance.type(torch.FloatTensor), dim=1)
            batch=inde.shape[0]

            # FDE
            dist=tolerance_rate  + fde[torch.arange(batch)[:,None],inde.view(-1,1)].squeeze()
            dist= dist/2.0

            tolerance_rate = tolerance_rate.unsqueeze(1).cuda()#dist.unsqueeze(1).cuda()

            # controller
            writing_prob =torch.sigmoid(self.linear_controller(tolerance_rate))#

            # future encoding
            future = torch.transpose(future, 1, 2)
            future_embed = self.relu(self.conv_fut(future))
            future_embed = torch.transpose(future_embed, 1, 2)
            output_fut, state_fut = self.encoder_fut(future_embed)

            index_writing = np.where(writing_prob.cpu() > 0.65)[0]#0.63  5622;0.5622

            batch_size = past_org.shape[0]
            index_writing2 = np.where(writing_prob.cpu() > 0.65)[0]#np.random.randint(batch_size, size=int(batch_size / 20.0))

            past_to_write = state_past.squeeze(0)[index_writing]
            future_to_write = state_fut.squeeze(0)[index_writing]
            #print(pred_gt.shape,state_past.shape)[248, 12, 2]) torch.Size([1, 248, 48]
            pred_gt_write= pred_gt[index_writing2]
            past_gt_write = obs_traj[index_writing2]

            pred_rel_gt_write = future_org[index_writing2]
            past_rel_gt_write = past_org[index_writing2]

            if len(past_to_write)>=1  :

                self.memory_past = torch.cat((self.memory_past, past_to_write), 0)
                self.memory_fut = torch.cat((self.memory_fut, future_to_write), 0)
                self.pred_gt = torch.cat((self.pred_gt,pred_gt_write), 0)
                self.past_gt = torch.cat((self.past_gt, past_gt_write), 0)
                self.pred_rel_gt = torch.cat((self.pred_rel_gt, pred_rel_gt_write), 0)
                self.past_rel_gt = torch.cat((self.past_rel_gt, past_rel_gt_write), 0)


        else:
            return prediction

        return writing_prob, tolerance_rate

    def write_in_memory(self, past, future,obs_traj=None,pred_gt=None):
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
        present_temp = obs_traj[:, -1].unsqueeze(1)#past[:, -1].unsqueeze(1)

        # past temporal encoding
        past = torch.transpose(past, 1, 2)
        story_embed = self.relu(self.conv_past(past))
        story_embed = torch.transpose(story_embed, 1, 2)
        output_past, state_past = self.encoder_past(story_embed)

        # Cosine similarity and memory read
        past_normalized = F.normalize(self.memory_past, p=2, dim=1)
        state_normalized = F.normalize(state_past.squeeze(), p=2, dim=1)
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

        th=self.th

        tolerance_1s = torch.sum(distances[:, :, :3] < th*0.25, dim=2)
        tolerance_2s = torch.sum(distances[:, :, 3:6] < th*0.5, dim=2)
        tolerance_3s = torch.sum(distances[:, :, 6:9] < th*0.75, dim=2)
        tolerance_4s = torch.sum(distances[:, :, 9:12] < th, dim=2)
        tolerance = tolerance_1s + tolerance_2s + tolerance_3s + tolerance_4s

        fde = torch.sum(distances[:, :, -1].view(-1, 20, 1) >= 1.3 * th, dim=-1)
        fde = fde.type(torch.FloatTensor)
        tolerance = (tolerance_1s + tolerance_2s + tolerance_3s + tolerance_4s)
        tolerance = tolerance.type(torch.FloatTensor) / 12.0

        tolerance = 1.0 - tolerance.type(torch.FloatTensor)
        #tolerance = tolerance + fde
        #tolerance = tolerance / 2.0

        tolerance_rate, inde = torch.max(tolerance.type(torch.FloatTensor), dim=1)
        # toler = tolerance_rate.cuda() + fde[inde].cuda()
        batch = inde.shape[0]

        # FDE
        dist = tolerance_rate + fde[torch.arange(batch)[:, None], inde.view(-1, 1)].squeeze()
        dist = dist / 2.0

        # print(distances[:, :, :8] < 100*0.25,tolerance_rate)torch.Size([25294, 48])
        # final destination

        tolerance_rate = tolerance_rate.unsqueeze(1).cuda()#dist.unsqueeze(1).cuda()
        # controller
        writing_prob = torch.sigmoid(self.linear_controller(tolerance_rate))

        # future encoding
        future = torch.transpose(future, 1, 2)
        future_embed = self.relu(self.conv_fut(future))
        future_embed = torch.transpose(future_embed, 1, 2)
        output_fut, state_fut = self.encoder_fut(future_embed)

        index_writing = np.where(writing_prob.cpu() > 0.65)[0]  # 0.63  5622;0.5622

        batch_size = past_org.shape[0]
        index_writing2 = index_writing# np.random.randint(batch_size, size=int(batch_size / 20.0))

        past_to_write = state_past.squeeze()[index_writing]
        future_to_write = state_fut.squeeze()[index_writing]
        # print(pred_gt.shape,state_past.shape)[248, 12, 2]) torch.Size([1, 248, 48]
        pred_gt_write = pred_gt[index_writing2]
        past_gt_write = obs_traj[index_writing2]

        pred_rel_gt_write = future_org[index_writing2]
        past_rel_gt_write = past_org[index_writing2]

        self.memory_past = torch.cat((self.memory_past, past_to_write), 0)
        self.memory_fut = torch.cat((self.memory_fut, future_to_write), 0)
        self.pred_gt = torch.cat((self.pred_gt, pred_gt_write), 0)

        self.past_gt = torch.cat((self.past_gt, past_gt_write), 0)

        self.pred_rel_gt = torch.cat((self.pred_rel_gt, pred_rel_gt_write), 0)
        self.past_rel_gt = torch.cat((self.past_rel_gt, past_rel_gt_write), 0)

        return torch.cat((past_to_write,future_to_write),dim=1)


