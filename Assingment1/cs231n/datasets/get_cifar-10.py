import torchvision.datasets as dset
trainset=dset.CIFAR10(root='datasets_cifar10', train=True, transform=None, target_transform=None, download=True)
testset=dset.CIFAR10(root= 'datasets_cifar10_tset', train=False, transform=None, target_transform=None, download=True)
#recommand to download on the official website for python version
#url:http://www.cs.toronto.edu/~kriz/cifar.html
#remember to unzip every zip file until you can see the detailed directory on windows
#otherwise the path line in ipynb cannot find the dataset