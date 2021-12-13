import torch
import os

from .container import ModelContainer

class TrainUnit(object):

    def __init__(self,model,device,best_acc=0,start_epoch=0,ckpt_path=None):
        assert isinstance(model,ModelContainer),"TypeException in train.py line124" 

        self.model = model
        self.device = device

        self.best_acc = best_acc
        self.start_epoch = start_epoch
        self.ckpt_path = ckpt_path

    def train_per_epoch(self,epoch):
        self.model.backbone.train()
        
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx,(inputs,targets) in enumerate(self.model.train_dataloader):
            
            inputs,targets = inputs.to(self.device),targets.to(self.device)
            self.model.optimizer.zero_grad()
            outputs = self.model.backbone(inputs)
            
            loss = self.model.criterion(outputs,targets)
            loss.backward()
            self.model.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(self.model.train_dataloader),'Train... Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    def verify_per_epoch(self,epoch):
        self.model.backbone.eval()

        verify_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.model.verify_dataloader):
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model.backbone(inputs)

                loss = self.model.criterion(outputs, targets)
               
                verify_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                print(batch_idx, len(self.model.verify_dataloader),'Verify... Loss: %.3f | Acc: %.3f%% (%d/%d)' % (verify_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        # Save checkpoint.
        acc = 100.*correct/total
        if acc > self.best_acc:
            ckpt_dir = os.path.dirname(self.ckpt_path)
            if not os.path.isdir(ckpt_dir):
                os.mkdir(ckpt_dir)
            print('Saving..')
            state = {
                'model': self.model.backbone.state_dict(),
                'acc': acc,
                'epoch': epoch,
                }
            torch.save(state, self.ckpt_path)
            self.best_acc = acc

    def train_for_epochs(self,epochs):

        for epoch in range(epochs):
            print("\nEpoch : {}".format(epoch))
            self.train_per_epoch(epoch)
            self.verify_per_epoch(epoch)
            self.model.scheduler.step()

    def meta_learn(self):
        pass