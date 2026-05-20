"""Example demonstrating the computation of alpha complexes.

This simple example demonstrates how to use alpha complexes to change
the appearance of a point cloud, following the `TopologyLayer
<https://github.com/bruel-gabrielsson/TopologyLayer>`_ package.

This example is still a **work in progress**.
"""

from torch_topological.nn import CubicalComplex
from torch_topological.nn import SummaryStatisticLoss
from torch_topological.utils import SelectByDimension
from torch_topological.data import sample_from_unit_cube
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image, ImageOps
import scipy


#  ╭──────────────────────────────────────────────────────────╮
#  │ Data                                                     │
#  ╰──────────────────────────────────────────────────────────╯

#img = Image.open("img.jpg").convert("L").convert('1')
#img = ImageOps.invert(img).rotate(315)
#img = img.resize((img.width//4,img.height//4))
#img = np.array(img)
#img = torch.tensor(img,dtype=torch.float)



img = torch.zeros(100,100)
#img[20:80,45:55] = 255
img[20:80,30:40] = 255
img[20:80,60:70] = 255
img[70:80,30:70] = 255

xs = torch.linspace(1, 0, steps=100)
ys = torch.linspace(0, 1, steps=100)
X, Y = torch.meshgrid(xs,ys)

theta = torch.nn.Parameter(torch.tensor(1.57, dtype=torch.float), requires_grad=True)
vr = CubicalComplex(dim=2)
loss_fn = SummaryStatisticLoss(
    summary_statistic='total_persistence',
    p=1,
    q=0
)
opt = torch.optim.Adam([theta], lr=.01)
for idx in range(300):
    heights = X * torch.cos(theta) + Y * torch.sin(theta) 
    pd = vr(255 - img * heights)
    loss = loss_fn(pd)
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        print(loss.numpy(), theta.numpy())


with torch.no_grad():
    heights = X * torch.cos(theta) + Y * torch.sin(theta) 
    final_img = 255 - img * heights
    pd = vr(255 - img * heights)
    plt.imshow(final_img)
    plt.show()



for theta in torch.linspace(0, torch.pi, steps=400):
    heights = X * torch.cos(theta) + Y * torch.sin(theta) 
    print((255 - img * heights).min())
