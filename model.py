import torch

G = torch.nn.Sequential(
        # input: (batch_size, 100, 1, 1)
        torch.nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
        torch.nn.BatchNorm2d(512),
        torch.nn.ReLU(True),
        # output: (batch_size, 512, 4, 4)

        torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
        torch.nn.BatchNorm2d(256),
        torch.nn.ReLU(True),

        torch.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(True),

        torch.nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(True),

        torch.nn.ConvTranspose2d(64, 3, 5, 3, 1, bias=False),
        torch.nn.Tanh() # redistribute to [-1, 1]
        # output: (batch_size, 3, 96, 96)

        )

D = torch.nn.Sequential(
        # input: (batch_size, 3, 96, 96)
        torch.nn.Conv2d(3, 64, 5, 3, 1, bias=False),
        torch.nn.LeakyReLU(0.2, inplace=True),

        torch.nn.Conv2d(64, 128, 4, 2, 1, bias=False),
        torch.nn.BatchNorm2d(128),
        torch.nn.LeakyReLU(0.2, inplace=True),

        torch.nn.Conv2d(128, 256, 4, 2, 1, bias=False),
        torch.nn.BatchNorm2d(256),
        torch.nn.LeakyReLU(0.2, inplace=True),

        torch.nn.Conv2d(256, 512, 4, 2, 1, bias=False),
        torch.nn.BatchNorm2d(512),
        torch.nn.LeakyReLU(0.2, inplace=True),
        # output: (batch_size, 512, 4, 4)

        torch.nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        torch.nn.Sigmoid(),
        # output: (batch_size, 1, 1, 1)

        )

