import torch
import numpy as np
aa=torch.tensor([
0.4875,
0.4898,
0.5270,
0.4850,
0.6262,
0.5625,
0.6252,
0.5201,
0.5917,
0.5576,
])
import torch


ade=0
fde=0
a=sum(aa)

print()
print(aa.shape,a/len(aa))

