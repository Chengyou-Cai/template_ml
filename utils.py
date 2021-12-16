import torch
import random
import numpy as np
import torchvision as tv

from torch import nn,optim
from models.backbones import resnet
from datasets.dataloader import *

from apis.container import ModelContainer
from apis.controller import Controller

def set_random_seed(seed,deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic

    # if deterministic:
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

def pipeline():
    backbone = resnet.ResNet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        backbone.parameters(), 
        lr=0.1,
        momentum=0.9, 
        weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    model_container = ModelContainer(
        model_name="resnet18_cifar10",
        backbone=backbone,
        train_dataloader=get_train_dataloader(tv.datasets.CIFAR10),
        test_dataloader=get_test_dataloader(tv.datasets.CIFAR10),
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    contorller = Controller(model=model_container)
    contorller.set_configs(epochs=1,gpu_id=0)
    contorller.init_model()
    contorller.train_model()

if __name__=="__main__":
    set_random_seed(seed=0,deterministic=True)
    pipeline()