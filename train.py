import torch
from torch.autograd import Variable
from model import G, D
import numpy as np
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import tqdm
import torch.utils.data as Data
import os

dataset_dir = 'data/'
checkpoints_dir = 'checkpoints/'
batch_size = 768
num_epochs = 200
num_workers = 16
os.makedirs(checkpoints_dir, exist_ok=True)

transforms = T.Compose([
    T.Resize(96),
    T.CenterCrop(96),
    T.ToTensor(),
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
    ])

dataset = ImageFolder(dataset_dir, transform=transforms)

dataloader = Data.DataLoader(
    dataset = dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers,
    drop_last = True,
    )

map_location = lambda storage, loc: storage

G = G.cuda()
D = D.cuda()

# load
if os.path.exists(os.path.join(checkpoints_dir, 'G.pkl')):
    print('loading G...')
    G = torch.load(os.path.join(checkpoints_dir, 'G.pkl'))
if os.path.exists(os.path.join(checkpoints_dir, 'D.pkl')):
    print('loading D...')
    D = torch.load(os.path.join(checkpoints_dir, 'D.pkl'))

G_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

for epoch in range(num_epochs):
    for index, (image, _) in tqdm.tqdm(enumerate(dataloader)):
        if index % 1 == 0:
            real_image = Variable(image)
            real_image = real_image.cuda()

            noises = torch.reshape(Variable(torch.from_numpy((np.vstack(torch.randn(100) for _ in range(batch_size))))), [batch_size, 100, 1, 1])
            noises = noises.cuda()
            
            fake_image = G(noises).detach()

            fake_pred = D(fake_image)
            real_pred = D(real_image)

            D_loss = - torch.mean(torch.log(real_pred)) - torch.mean(torch.log(1 - fake_pred))

            D_optimizer.zero_grad()
            #D_loss.backward(retain_graph=True)
            D_loss.backward()
            D_optimizer.step()

        if index % 5 == 0: 
            noises = torch.reshape(Variable(torch.from_numpy((np.vstack(torch.randn(100) for _ in range(batch_size))))), [batch_size, 100, 1, 1])
            noises = noises.cuda()
            
            fake_image = G(noises)

            fake_pred = D(fake_image)

            G_loss = torch.mean(torch.log(1 - fake_pred))


            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()
            
        if index % 10 == 0:
            print('epoch {}: D loss={}; G loss={}'.format(epoch, D_loss, G_loss))
    
    # save
    torch.save(G, 'checkpoints/G.pkl')
    torch.save(D, 'checkpoints/D.pkl')
