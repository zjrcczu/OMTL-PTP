import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import torch
nm='zara1'
a=torch.load('/home/chen/disk1/newCLTPMAN_ONLINE/CLTPMAN_ONLINE_fan/traj/random/randomorder_'+str(nm) +'ft.pt')
awo=torch.load('/home/chen/disk1/newCLTPMAN_ONLINE/CLTPMAN_ONLINE_fan/traj/random_wo/randdom_'+str(nm) +'wo.pt')
past=a['traj'][:,:8]
future=a['traj'][:,8:]
pred_best=a['pred']
pred_wo_best=awo['pred']
print(past.shape)
pic_cnt=0
path='./zara1_wo'
import os
if not os.path.exists(path):
    os.mkdir(path)


for i in range(past.shape[0]):

    plt.figure(figsize=(20, 15), dpi=100)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    ground_truth_input_x_piccoor = (
        past[i, :, :].cpu().numpy()[:,  0]#.T
    )
    ground_truth_input_y_piccoor = (
        past[i, :, :].cpu().numpy()[:,  1]#.T bs,8,2
    )
    ground_truth_output_x_piccoor = (
        future[i,  :,:].cpu().numpy()[:,  0]#.T
    )
    ground_truth_output_y_piccoor = (
        future[i, :, :].cpu().numpy()[:, 1]#.T
    )
    model_output_x_piccoor = (
        pred_best[i, :,:].cpu().detach().numpy()[:,  0]#.T
    )
    model_output_y_piccoor = (
        pred_best[i, :, :].cpu().detach().numpy()[:,  1]#.T
    )
    output_x_piccoor = (
        pred_wo_best[i, :, :].cpu().detach().numpy()[:, 0]  # .T
    )
    output_y_piccoor = (
        pred_wo_best[i, :, :].cpu().detach().numpy()[:, 1]  # .T
    )



    observed_line = plt.plot(
        ground_truth_input_x_piccoor[ :],
        ground_truth_input_y_piccoor[ :],
        "r-",
        linewidth=4,
        label="Observed Trajectory",
    )[0]
    observed_line.axes.annotate(
        "",
        xytext=(
            ground_truth_input_x_piccoor[ -2],
            ground_truth_input_y_piccoor[ -2],
        ),
        xy=(
            ground_truth_input_x_piccoor[ -1],
            ground_truth_input_y_piccoor[ -1],
        ),
        arrowprops=dict(
            arrowstyle="->", color=observed_line.get_color(), lw=4
        ),
        size=20,
    )
    ground_line = plt.plot(
        np.append(
            ground_truth_input_x_piccoor[ -1],
            ground_truth_output_x_piccoor[ :],
        ),
        np.append(
            ground_truth_input_y_piccoor[-1],
            ground_truth_output_y_piccoor[ :],
        ),
        "b-",
        linewidth=4,
        label="Ground Truth",
    )[0]


    plt.scatter(ground_truth_input_x_piccoor[ -1], ground_truth_input_y_piccoor[-1], s=300, c='b', marker='*')
    predict_line = plt.plot(
        np.append(
            ground_truth_input_x_piccoor[ -1],
            model_output_x_piccoor[ :],
        ),
        np.append(
            ground_truth_input_y_piccoor[ -1],
            model_output_y_piccoor[ :],
        ),
        color="g",  # ffff00
        ls="--",
        linewidth=4,
        label="Predicted Trajectory",
    )[0]
    predict_line2 = plt.plot(
        np.append(
            ground_truth_input_x_piccoor[-1],
            output_x_piccoor[:],
        ),
        np.append(
            ground_truth_input_y_piccoor[-1],
            output_y_piccoor[:],
        ),
        color="m",  # ffff00
        ls="--",
        linewidth=4,
        label="Predicted Trajectory",
    )[0]

    # plt.axis("off")
    plt.savefig(
       path+'/pic_{}.png'.format(pic_cnt)
    )
    plt.close()
    pic_cnt += 1
    print(pic_cnt)
