import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

torch.manual_seed(42)

def get_augmented_data(data_name, batch_size=256, size=32, s=1.0):
    k_size= int(0.1*size)
    self.batch_size = batch_size
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    blur = transforms.GaussianBlur((k_size, k_size), sigma=(0.1, 2.0))

    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.RandomApply([blur], p=0.5),
                                          transforms.ToTensor()])
    if data_name == 'CIFAR10':
        trainset_t0 = datasets.CIFAR10(root='/home/mmesarcik/data', train=True, download=True, transform=data_transforms)
        trainloader_t0 = DataLoader(trainset_t0, batch_size=self.batch_size, shuffle=False, num_workers=2)

        trainset_t1 = datasets.CIFAR10(root='/home/mmesarcik/data', train=True, download=True, transform=data_transforms)
        trainloader_t1 = DataLoader(trainset_t1, batch_size=self.batch_size, shuffle=False, num_workers=2)

        testset = datasets.CIFAR10(root='/home/mmesarcik/data', train=False, download=True, transform=data_transforms)
        testloader = DataLoader(testset, batch_size=self.batch_size,shuffle=False, num_workers=2)

    elif data_name == 'FMNIST':
        trainset_t0 = datasets.FashionMNIST(root='/home/mmesarcik/data', train=True, download=True, transform=data_transforms)
        trainloader_t0 = DataLoader(trainset_t0, batch_size=self.batch_size, shuffle=False, num_workers=2)

        trainset_t1 = datasets.FashionMNIST(root='/home/mmesarcik/data', train=True, download=True, transform=data_transforms)
        trainloader_t1 = DataLoader(trainset_t1, batch_size=self.batch_size, shuffle=False, num_workers=2)

        testset = datasets.FashionMNIST(root='/home/mmesarcik/data', train=False, download=True, transform=data_transforms)
        testloader = DataLoader(testset, batch_size=self.batch_size,shuffle=False, num_workers=2)

    return trainloader_t0, trainloader_t1, testloader

