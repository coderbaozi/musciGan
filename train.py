import torch
from generator import Generator
from discriminator import Discriminator
from model import Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

netG = Generator().to(device)
netD = Discriminator().to(device)
model = Model(netG, netD, '/mlx_devbox/users/qianziyang/playground/musciGan/datasets/*/*.wav', device)
model.train(num_epochs=100)