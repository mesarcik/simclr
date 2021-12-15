import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from torch.nn import CosineSimilarity

from tqdm import tqdm

from data import get_augmented_data
from model import Resnet

_epochs=100
_batch_size=256
_device = None
_gpu =0 
_t=0.07

_sim = CosineSimilarity(dim=1, eps=1e-6)


def simclr_loss(negative_z_0, negative_z_1, positive_z_0, positive_z_1):
    positive = _sim(positive_z_0, positive_z_1)
    negative = 0
    for i in range(_batch_size-1):
        negative += torch.exp(_sim(negative_z_0[i:i+1,:],
                                   negative_z_1[i:i+1,:])/_t).item()

    loss = -torch.log(torch.exp(positive/_t)/negative)
    return loss

def main():
    if torch.cuda.is_available():
        _device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True

    model = Resnet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003,weight_decay= 1e-4)

    trainloader_t0, trainloader_t1, testloader = get_augmented_data('CIFAR10', 
                                                                    batch_size=_batch_size,
                                                                    size=32, 
                                                                    s=1.0)
    model.cuda()
    with torch.cuda.device(_gpu):
        for epoch in range(_epochs):
            for x_0, x_1 in tqdm(zip(trainloader_t0, trainloader_t1),unit='batch'):

                x_0, x_1 = x_0[0], x_1[0]# remove labels 
                z_0 = model(x_0.to(_device))
                z_1 = model(x_1.to(_device))
                loss = 0
                for i in range(_batch_size):

                    # the positive pair of augmented 
                    positive_0= x_0[i:i+1,...]
                    positive_1 = x_1[i:i+1,...]

                    positive_z_0 = z_0[i:i+1,...]
                    positive_z_1 = z_1[i:i+1,...]

                    # the negative pair of augmented 
                    negative_0 = torch.cat((x_0[:i,...],x_0[i+1:,...]), axis=0)
                    negative_1 = torch.cat((x_1[:i,...],x_1[i+1:,...]), axis=0)

                    negative_z_0 = torch.cat((z_0[:i,...],z_0[i+1:,...]), axis=0)
                    negative_z_1 = torch.cat((z_1[:i,...],z_1[i+1:,...]), axis=0)
                    loss += simclr_loss(negative_z_0, negative_z_1, positive_z_0, positive_z_1)

                loss = loss/2*_batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('loss = {}'.format(loss.item()))

if __name__ == '__main__':
    main()
