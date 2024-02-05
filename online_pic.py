import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import torch
a=torch.load('zara2_wo.pt')
awo=torch.load('eth.pt')
past=a['traj'][:,:8]

print(past.shape)
pic_cnt=0
