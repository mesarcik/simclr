import numpy as np
import os 
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from data import get_data
from model import Resnet
from utils import _colors, _cifar10_labels, load_checkpoint

_batch_size =1024
_gpu=0

def vis_embeddings(model_dir, z_test, z_test_labels, limit=2000):
    z_embedded = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(z_test)

    for i in range(limit):
        plt.scatter(z_embedded[i,0], 
                    z_embedded[i,1], 
                    color = _colors[int(z_test_labels[i])],
                    label = _cifar10_labels[int(z_test_labels[i])]);
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig(os.path.join(model_dir,'embeding'), dpi=300)

def linear_classifcation(model_dir,limit=20000):
    if torch.cuda.is_available():
        _device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True

    # load model
    model = load_checkpoint(Resnet())

    # remove last layer 
    model =torch.nn.Sequential(*list(list(model.children())[0].children())[:-1])

    # get z vector with training class
    trainloader, testloader = get_data('CIFAR10', 
                                       augment=False, 
                                       batch_size=_batch_size,
                                       size=32,  
                                       s=1.0)
    model.cuda()
    with torch.cuda.device(_gpu):
        for i,(image_batch, labels_batch) in enumerate(trainloader):
            if i ==0:
                z_train = model(image_batch.to(_device))
                z_train_labels = labels_batch 
            else:
                z_train = torch.cat((z_train, model(image_batch.to(_device))), axis=0)
                z_train_labels = torch.cat((z_train_labels, labels_batch ), axis=0)

        for i,(image_batch, labels_batch) in enumerate(testloader):
            if i ==0:
                z_test = model(image_batch.to(_device))
                z_test_labels = labels_batch 
            else:
                z_test = torch.cat((z_test, model(image_batch.to(_device))), axis=0)
                z_test_labels = torch.cat((z_test_labels, labels_batch ), axis=0)
        
    z_train = z_train.cpu().detach().numpy()[:limit,...,0,0]
    z_train_labels = z_train_labels.cpu().detach().numpy()[:limit]
    z_test = z_test.cpu().detach().numpy()[:limit,...,0,0]
    z_test_labels = z_test_labels.cpu().detach().numpy()[:limit]

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(z_train, z_train_labels)
    y = svclassifier.predict(z_test)

    print(confusion_matrix(z_test_labels,y))
    print(classification_report(z_test_labels,y))
    vis_embeddings(model_dir,z_test, z_test_labels)




