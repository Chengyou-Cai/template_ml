import os,argparse
import random
import numpy as np
import torch
from torch import nn,optim

import torchvision as tv

from datasets.dataloader import *
from models.backbones import resnet
from apis.train import Trainer

class Pipeline(object):
    
    def __init__(self,args,model_name,model,best_acc=0):
        self.args = args
        self.model_name = model_name #"cifar10_resnet18"
        self.model = model
        self.best_acc = best_acc
        
        self.dataset = tv.datasets.CIFAR10
        self.training_dataloader = get_training_dataloader(self.dataset)
        self.test_dataloader = get_test_dataloader(self.dataset)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=0.1,
            momentum=0.9, 
            weight_decay=5e-4
            )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        self.trainer = Trainer(
            self.model_name,
            self.model,
            self.best_acc,
            self.args.device,
            self.training_dataloader,
            self.test_dataloader,
            self.criterion,
            self.optimizer,
            self.scheduler
            )

    def train(self):
        
        self.trainer.train_epochs(self.args.epochs)

    def test(self):
        pass

def set_random_seed(seed,deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic

    # if deterministic:
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser("Init Argument")

    parser.add_argument('--gpuid',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=50)

    args = parser.parse_args()

    args.device = torch.device("cuda:{}".format(args.gpuid) if torch.cuda.is_available() else "cpu")

    return args     


if __name__=="__main__":
    set_random_seed(seed=0,deterministic=True)

    args = get_args()
    print(args)

    pipeline = Pipeline(args=args,model_name="cifar10_resnet18",model=resnet.ResNet18())
    pipeline.train()