import torch
import torch.nn as nn
import torch.optim as optim

import os
from pathlib import Path

class Trainer(object):

    def __init__(
        self,
        model_name,
        model,
        best_acc,
        device,
        training_dataloader,
        validation_dataloader,
        criterion,
        optimizer,
        scheduler
        ):

        assert isinstance(model,nn.Module),"ModelException" 
        self.model_name = model_name
        self.model = model
        
        self.device = device
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.best_acc = best_acc

        self.checkpoint_dir = 'checkpoint_'+self.model_name
        self.init_model()

    def  init_model(self):
        if self.device == "cuda":
            self.model = nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = True
        
        p = Path('./datasets/{}/acc_ckpt.pth'.format(self.checkpoint_dir))
        if Path.exists(p):
            ckpt = torch.load(p)
            self.best_acc = ckpt['acc']
            self.model.load_state_dict(ckpt['model'])
        else:
            print("warning: model path doesn't exists")

    def train_per_epoch(self,epoch):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx,(inputs,targets) in enumerate(self.training_dataloader):
            inputs,targets = inputs.to(self.device),targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            loss = self.criterion(outputs,targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(self.dataloader),'Train... Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    def verify_per_epoch(self,epoch):
        self.model.eval()

        verify_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.validation_dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                verify_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                print(batch_idx, len(self.validation_dataloader),'Verify... Loss: %.3f | Acc: %.3f%% (%d/%d)' % (verify_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        # Save checkpoint.
        acc = 100.*correct/total
        state = {
            'model': self.model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        torch.save(state, './{}/epoch_{:0>5}.pth'.format(self.checkpoint_dir,epoch))

        if acc > self.best_acc:
            print('Saving..')
            torch.save(state, './{}/acc_ckpt.pth'.format(self.checkpoint_dir))
            self.best_acc = acc

    def train_epochs(self,epochs):

        for epoch in range(epochs):
            print("\nEpoch : {}".format(epoch))
            self.train_per_epoch(epoch)
            self.verify_per_epoch(epoch)
            self.scheduler.step()

    def meta_learning_train(self):
        pass


# def train(device,model,epochs,dataloader,cfgs):
#     assert isinstance(model,torch.nn.Module),"ModelException"
#     model.train()

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(
#         model.parameters(), 
#         lr=cfgs.lr,
#         momentum=0.9, 
#         weight_decay=5e-4
#         )
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
#     for epoch in range(epochs):

#         print("\nEpoch : {}".format(epoch))
#         train_loss = 0
#         correct = 0
#         total = 0
        
#         for batch_idx,(inputs,targets) in enumerate(dataloader):
#             inputs,targets = inputs.to(device),targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs,targets)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#             print(batch_idx, len(dataloader),'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# def meta_learning_train():
#     pass