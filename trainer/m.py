'''
config = self.config
self.mem_n2n.train()
self.mem_n2n_train.eval()

base_weights = torch.load(config.mantra_model)
base_weights = list(base_weights.parameters(
))
"""
     Training loop over the dataset for an epoch
     :return: loss
     """
config = self.config
self.mem_n2n.train()
self.mem_n2n_train.eval()

base_weights = torch.load(config.mantra_model)
base_weights = list(base_weights.parameters(
))
# parameters that need not be decayed


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

    self.opt.zero_grad()
    ### old data
    losst = torch.zeros(1).to(pred_traj_gt).cuda()
    l2_loss_rel_train = []
    past_train, future_train, past_rel_train, future_rel_train = self.replay_memory()  # self.replay_init(self.data_train)

    past_train = Variable(past_train).cuda()

    future_train = Variable(future_train).cuda()

    past_rel_train = Variable(past_rel_train).cuda()

    future_rel_train = Variable(future_rel_train).cuda()

    output_train = self.mem_n2n(past_rel_train, obs_traj=past_train)
    # args.pred_len, mem, epoch,
    l2_loss_sum_rel_train = torch.zeros(1).to(pred_traj_gt).cuda()
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

    l2_loss_sum_rel_train += torch.mean(best).reshape(1).cuda()

    losst = losst + l2_loss_sum_rel_train

    # Current model weights
    curr_weights = list(self.mem_n2n.parameters())
    # new data
    output = self.mem_n2n(past_rel, obs_traj=past)

    loss = torch.zeros(1).to(pred_traj_gt).cuda()
    l2_loss_rel = []
    loss_mask = loss_mask[:, config.obs_len:]

    # args.pred_len, mem, epoch,
    for topki in range(config.best_k):
        pred_traj_fake = output[:, topki]  # ([729, 20, 12, 2])
        # pred_traj_fake_rel = pred_traj_fake_rel[-self.pred_len:]
        pred_traj_fake = pred_traj_fake.transpose(1, 0)
        l2_loss_rel.append(
            l2_loss(pred_traj_fake, pred_traj_gt, loss_mask, mode="raw")
        )

    l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt).cuda()
    l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
    for start, end in seq_start_end.data:
        _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
        _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
        _l2_loss_rel = torch.min(_l2_loss_rel) / (
                (pred_traj_fake.shape[0]) * (end - start)
        )
        l2_loss_sum_rel = l2_loss_sum_rel + _l2_loss_rel

    loss = loss + 0.7 * l2_loss_sum_rel + 0.3 * losst + 0.001 * diff_loss
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(self.mem_n2n.parameters(), 1.0, norm_type=2)
    self.opt.step()

    self.writer.add_scalar('loss/loss_total', loss, self.iterations)
print('ff', losst)

return loss.item()
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

    self.opt.zero_grad()
    output = self.mem_n2n(past_rel, obs_traj=past)

    loss = torch.zeros(1).to(pred_traj_gt).cuda()
    l2_loss_rel = []
    loss_mask = loss_mask[:, config.obs_len:]

    # args.pred_len, mem, epoch,
    for topki in range(config.best_k):
        pred_traj_fake = output[:, topki]  # ([729, 20, 12, 2])
        # pred_traj_fake_rel = pred_traj_fake_rel[-self.pred_len:]
        pred_traj_fake = pred_traj_fake.transpose(1, 0)
        l2_loss_rel.append(
            l2_loss(pred_traj_fake, pred_traj_gt, loss_mask, mode="raw")
        )

    l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt).cuda()
    l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
    for start, end in seq_start_end.data:
        _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
        _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
        _l2_loss_rel = torch.min(_l2_loss_rel) / (
                (pred_traj_fake.shape[0]) * (end - start)
        )
        l2_loss_sum_rel = l2_loss_sum_rel + _l2_loss_rel


    l2_loss_rel_train = []
    past_train, future_train, past_rel_train, future_rel_train = self.replay_memory()  # self.replay_init(self.data_train)

    past_train = Variable(past_train).cuda()

    future_train = Variable(future_train).cuda()

    past_rel_train = Variable(past_rel_train).cuda()

    future_rel_train = Variable(future_rel_train).cuda()

    output_train = self.mem_n2n(past_rel_train, obs_traj=past_train)
    # args.pred_len, mem, epoch,
    l2_loss_sum_rel_train = torch.zeros(1).to(pred_traj_gt).cuda()
    rmse_all = []
    for topki in range(config.best_k):
        pred_traj_fake_train = output_train[:, topki]  # ([729, 20, 12, 2])
        # pred_traj_fake_rel = pred_traj_fake_rel[-self.pred_len:]
        # .transpose(1, 0)

        # rmse
        batch, seq_len, _ = pred_traj_fake_train.size()
        # equation below , the first part do noing, can be delete
        loss_f = (pred_traj_fake_train - future_train) ** 2
        rmse = (loss_f.sum(dim=2).sum(dim=1) / seq_len).reshape(1, -1)

        rmse_all.append(rmse)
    rmse_ = torch.stack(rmse_all, dim=0).squeeze()

    best, _ = torch.min(rmse_, dim=0)

    l2_loss_sum_rel_train += torch.mean(best).reshape(1).cuda()

    loss = loss + 5.0 * l2_loss_sum_rel_train + l2_loss_sum_rel

    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.mem_n2n.parameters(), 1.0, norm_type=2)
    self.opt.step()

    self.writer.add_scalar('loss/loss_total', loss, self.iterations)
print('ff', l2_loss_sum_rel_train)

return loss.item()


def _train_single_epoch(self):
    """
    Training loop over the dataset for an epoch
    :return: loss
    """
    config = self.config
    torch.backends.cudnn.enabled = False
    self.mem_n2n_train.eval()

    base_weights = torch.load(config.mantra_model)
    base_weights = list(base_weights.parameters(
    ))
    aa = []

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
        self.mem_n2n_train.memory_past = torch.load('./pretrained_models/MANTRA/increm_mem/memory_past.pt')
        self.mem_n2n_train.memory_fut = torch.load('./pretrained_models/MANTRA/increm_mem/memory_fut.pt')

        pred_traj_fake_gt_org = self.mem_n2n_train(past_rel, obs_traj=past)

        loss = torch.zeros(1).to(pred_traj_gt).cuda()
        for j in range(1):
            self.opt.zero_grad()
            output = self.mem_n2n(past_rel, obs_traj=past)

            loss = torch.zeros(1).to(pred_traj_gt).cuda()

            l2_loss_sum_rel_train = 0
            # equation below , the first part do noing, can be delete
            for topki in range(config.best_k):
                pred_traj_fake_train = pred_traj_fake_gt_org[:, topki]  # ([729, 20, 12, 2])
                # pred_traj_fake_rel = pred_traj_fake_rel[-self.pred_len:]
                # .transpose(1, 0)

                # rmse
                batch, seq_len, _ = pred_traj_fake_train.size()
                # equation below , the first part do noing, can be delete
                loss_f = (pred_traj_fake_train - output[:, topki]) ** 2
                rmse = (loss_f.sum(dim=2).sum(dim=1) / seq_len)
                l2_loss_sum_rel_train += torch.mean(rmse)

            curr_weights = list(self.mem_n2n.parameters())
            diff_loss = torch.Tensor([0]).cuda()
            # Iterate over base_weights and curr_weights and accumulate the euclidean norm
            # of their differences
            for base_param, curr_param in zip(base_weights, curr_weights):
                diff_loss += (curr_param - base_param).pow(2).sum()

            loss = loss + 0.001 * diff_loss + 0.1 * l2_loss_sum_rel_train

            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.mem_n2n.parameters(), 1.0, norm_type=2)
            self.opt.step()


import torch
import numpy as np
from collections import Counter

if __name__ == '__main__':

    a = [
        [2, 3, 34, 56],
        [14, 23, 45, 6],
        [2, 3, 34, 56]
    ]

    map = {}

    for item in a:
        s = str(item)
        if s in map.keys():
            map[s] = map[s] + 1
        else:
            map[s] = 1

    for key in map.keys():
        print('%s的次数为%d' % (key, map[key]))

'''