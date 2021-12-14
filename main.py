import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from torch.nn import CosineSimilarity as sim

from tqdm import tqdm

from data import get_augmented_data
from model import Resnet

_batch_size=256
_device = None
_gpu =0 
_t=0.07

def simclr_loss(z_0, z_1, z_i, z_j):
    positive = sim(z_i, z_j)
    negative = 0
    for i,j in zip(z_0, z_1):
        negative += torch.exp(sim(i,j)/_t)

    loss = -torch.log(torch.exp(positive/_t)/negative)
    return loss

def main():
    if torch.cuda.is_available():
        print('here')
        _device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True

    model = Resnet().to(_device)
    optimizer = torch.optim.Adam(model.parameters(), 0.0003, 1e-4)

    trainloader_t0, trainloader_t1, testloader = get_augmented_data('CIFAR10', 
                                                                    batch_size=_batch_size,
                                                                    size=32, 
                                                                    s=1.0)
    with torch.cuda.device(_gpu):
        for x_0, x_1 in zip(trainloader_t0, trainloader_t1):
            for i in range(_batch_size):
                x_0, x_1 = x_0[0], x_1[0]# remove labels 

                positive_0= x_0[i:i+1,...].to(_device)# the positive pair of augmented 
                positive_1 = x_1[i:i+1,...].to(_device)
                z_i = model(positive_0)
                z_j = model(positive_1)

                negative_0 = torch.cat((x_0[:i,...],# the negative pair of augmented 
                                      x_0[i+1:,...]), axis=0).to(_device)
                negative_1 = torch.cat((x_1[:i,...], 
                                      x_1[i+1:,...]), axis=0).to(_device) 

                z_0 = model(negative_0)
                z_1 = model(negative_1)

                loss = simclr_loss(z_0, z_1, z_i, z_j) 
                  
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('loss = {}'.format(loss))





if __name__ == '__main__':
    main()
