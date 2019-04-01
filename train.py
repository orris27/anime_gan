import torch
from torch.autograd import Variable
from model import G, D
import numpy as np
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import tqdm
import torch.utils.data as Data
import os

#####################################################################
# Constant
#####################################################################
dataset_dir = 'data/'
checkpoints_dir = 'checkpoints/'
batch_size = 768
num_epochs = 200
num_workers = 16
os.makedirs(checkpoints_dir, exist_ok=True)

#####################################################################
# Data Loader
#####################################################################
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

true_labels = torch.ones(batch_size).cuda() # not Variable
fake_labels = torch.zeros(batch_size).cuda()

noises = torch.randn(batch_size, 100, 1, 1)
noises = noises.cuda()

#####################################################################
# Create Discriminator & Generator
#####################################################################
G = G.cuda()
D = D.cuda()

if os.path.exists(os.path.join(checkpoints_dir, 'G.pkl')):
    print('loading G...')
    G = torch.load(os.path.join(checkpoints_dir, 'G.pkl'))
if os.path.exists(os.path.join(checkpoints_dir, 'D.pkl')):
    print('loading D...')
    D = torch.load(os.path.join(checkpoints_dir, 'D.pkl'))


#####################################################################
# Loss & Optimizer
#####################################################################
loss_fn = torch.nn.BCELoss()

G_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

for epoch in range(num_epochs):
    for index, (image, _) in tqdm.tqdm(enumerate(dataloader)):
        real_image = Variable(image)
        real_image = real_image.cuda()


        #####################################################################
        # Discriminator
        #####################################################################
        if index % 1 == 0:
            D_optimizer.zero_grad()
            
            real_pred = D(real_image)
            D_loss_real = loss_fn(real_pred, true_labels)

            noises.data.normal_()
            fake_image = G(noises).detach() # fake_image.requires_grad => False
            fake_pred = D(fake_image)
            D_loss_fake = loss_fn(fake_pred, fake_labels)

            D_loss = D_loss_real + D_loss_fake
            D_loss.backward()
            D_optimizer.step()

        #####################################################################
        # Generator
        #####################################################################
        if index % 5 == 0: 
            G_optimizer.zero_grad()
            
            noises.data.normal_()
            fake_image = G(noises)
            fake_pred = D(fake_image)
            G_loss = loss_fn(fake_pred, true_labels)
            G_loss.backward()
            G_optimizer.step()
            
        if index % 10 == 0:
            print('epoch {}: D loss={}; G loss={}'.format(epoch, D_loss, G_loss))
    
    #####################################################################
    # Save Model
    #####################################################################
    torch.save(G, 'checkpoints/G.pkl')
    torch.save(D, 'checkpoints/D.pkl')
