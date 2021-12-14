# adapted from https://github.com/sthalles/SimCLR/blob/master/models/resnet_simclr.py

import torch.nn as nn
import torchvision.models as models

class Resnet(nn.Module):
    def __init__(self, out_dim=128):
        super(Resnet, self).__init__()

        resnet = models.resnet18(pretrained=False, num_classes=out_dim)
        self.dim_mlp = resnet.fc.in_features
        self.temp_fc = resnet.fc
        
        modules=list(resnet.children())[:-1]
        self.resnet =nn.Sequential(*modules)

        self.fc = nn.Sequential(nn.Linear(self.dim_mlp, self.dim_mlp), nn.ReLU(), self.temp_fc)

    def forward(self, x):
        out = self.resnet(x)
        return self.fc(out)
    
    def project(self,x):
        return self.resnet(x)
