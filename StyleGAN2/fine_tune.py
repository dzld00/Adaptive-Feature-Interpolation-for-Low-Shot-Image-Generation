#import hydra
#from omegaconf import DictConfig
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10
import torchvision
from torchvision import transforms
#from torchvision.models import resnet18, resnet34, resnet50
#from models import SimCLR
from tqdm import tqdm

#from resnet import Resnet18_ML, Resnet34_ML
from training.networks import Discriminator


logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LinModel(nn.Module):
    """Linear wrapper of encoder."""
    def __init__(self, feature_dim: int, n_classes: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.lin = nn.Linear(self.feature_dim, self.n_classes)

    def forward(self, x):
        x = F.normalize(x)
        return self.lin(x)


def run_epoch(pre_model, model, dataloader, epoch, optimizer=None, scheduler=None):
    if optimizer:
        model.train()
    else:
        model.eval()

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    loader_bar = tqdm(dataloader)
    
    pre_model.eval()
    for x, y in loader_bar:
        x, y = x.cuda(), y.cuda()
        
        fea = pre_model(img=x,c=0)
        logits = model(fea)
        loss = F.cross_entropy(logits, y)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        acc = (logits.argmax(dim=1) == y).float().mean()
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc.item(), x.size(0))
        if optimizer:
            loader_bar.set_description("Train epoch {}, loss: {:.4f}, acc: {:.4f}"
                                       .format(epoch, loss_meter.avg, acc_meter.avg))
        else:
            loader_bar.set_description("Test epoch {}, loss: {:.4f}, acc: {:.4f}"
                                       .format(epoch, loss_meter.avg, acc_meter.avg))

    return loss_meter.avg, acc_meter.avg


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


#@hydra.main(config_path='simclr_config.yml')
def finetune():
    NUM_EPOCHS = 100
    learning_rate = 1 #biggan/resnet:0.1 #restnet18_ml/netML:10 #default:0.6
    batch_size = 512
    feature_dim = 1024 #1024 #512
    n_classes = 10
        
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor()])
    test_transform = transforms.ToTensor()
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        ])
    
    data_dir = '/data/mendai/cifar10'

    trainset = torchvision.datasets.CIFAR10(root='/data/mendai/cifar10', train=True,
                                            download=True, transform=transform)
    indices = np.random.choice(len(trainset), 10*n_classes, replace=False)
    sampler = SubsetRandomSampler(indices)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              drop_last=True)
    testset = torchvision.datasets.CIFAR10(root='/data/mendai/cifar10', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False,drop_last=True)
    print(len(train_loader))
    print(len(test_loader))
    
    import pickle
    path_to_checkpoint = '/vc_data/mendai/mendai/data/MM/results/cifar10_s2-ada/00035-cifar10-cifar-gamma1e-06-batch128-noaug/network-snapshot-012288.pkl'
    with open(path_to_checkpoint, 'rb') as f:
        data = pickle.load(f)
    pre_model = data['D'].to(device)
    model = LinModel(feature_dim=feature_dim, n_classes=10)
    model = model.to(device)
    
    print("Encoder:")
    print(pre_model)
    print("Linear:")
    print(model)

    # Fix encoder
    pre_model.requires_grad = False
    parameters = [param for param in model.parameters() if param.requires_grad is True]  # trainable parameters.
    
    #optimizer = Adam(parameters, lr=0.01)

    optimizer = torch.optim.SGD(
        parameters,
        0.2, #0.2   # lr = 0.1 * batch_size / 256, see section B.6 and B.7 of SimCLR paper.
        momentum=0.9,
        weight_decay=0.,
        nesterov=True)
    

    # cosine annealing lr
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            NUM_EPOCHS * len(train_loader),
            learning_rate,  # lr_lambda computes multiplicative factor
            1e-4)) #1e-4, 1e-3

    optimal_loss, optimal_acc = 1e5, 0.
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(pre_model, model, train_loader, epoch, optimizer, scheduler)
        test_loss, test_acc = run_epoch(pre_model, model, test_loader, epoch)

        if train_loss < optimal_loss:
            optimal_loss = train_loss
            optimal_acc = test_acc
            logger.info("==> New best results")
            #torch.save(model.state_dict(), 'simclr_lin_{}_best.pth'.format(args.backbone))

    logger.info("Best Test Acc: {:.4f}".format(optimal_acc))
    print("optimal accuracy: ", optimal_acc)


if __name__ == '__main__':
    finetune()