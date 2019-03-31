import torch
from torch.autograd import Variable
from model import G, D
import numpy as np
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import torchvision
import tqdm
import torch.utils.data as Data
import os

checkpoints_dir = 'checkpoints/'
batch_size = 256

# load
if os.path.exists(os.path.join(checkpoints_dir, 'G.pkl')):
    print('loading G...')
    G = torch.load(os.path.join(checkpoints_dir, 'G.pkl'))
if os.path.exists(os.path.join(checkpoints_dir, 'D.pkl')):
    print('loading D...')
    D = torch.load(os.path.join(checkpoints_dir, 'D.pkl'))

G.eval()
D.eval()

noises = torch.reshape(Variable(torch.from_numpy((np.vstack(torch.randn(100) for _ in range(batch_size))))), [batch_size, 100, 1, 1])
noises = noises.cuda()

fake_image = G(noises)

scores = D(fake_image).detach()
scores = scores.data.squeeze()
indexs = scores.topk(64)[1]
result = list()
for index in indexs:
    result.append(fake_image.data[index])

torchvision.utils.save_image(torch.stack(result), 'result.png', normalize=True, range=(-1, 1))
