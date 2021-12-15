# adapted from https://github.com/sthalles/SimCLR/blob/master/models/resnet_simclr.py

import torch.nn as nn
import torchvision.models as models

class Resnet(nn.Module):
    def __init__(self, out_dim=128):
        super(Resnet, self).__init__()

        self.resnet = models.resnet18(pretrained=False, num_classes=out_dim)
        dim_mlp = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.resnet.fc)


    def forward(self, x):
        return self.resnet(x)
    
    #def project(self,x):
    #    return self.resnet(x)
