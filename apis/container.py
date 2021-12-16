from torch import nn
import torch
from torch.utils.data import DataLoader

class ModelContainer(object):

    def __init__(
        self,
        model_name='default_backbone_dataset',
        backbone=None,
        train_dataloader=None,
        verify_dataloader=None,
        test_dataloader=None,
        criterion=None,
        optimizer=None,
        scheduler=None,
        ):
        assert isinstance(backbone,nn.Module),"backbone isn't an instance of torch.nn.Module"
        
        self.model_name = model_name
        self.backbone = backbone
        
        self.train_dataloader = train_dataloader
        self.verify_dataloader = verify_dataloader if verify_dataloader else test_dataloader
        self.test_dataloader = test_dataloader 

        if self.train_dataloader:
            assert isinstance(self.train_dataloader,DataLoader)
        if self.verify_dataloader:
            assert isinstance(self.verify_dataloader,DataLoader)
        if self.test_dataloader:
            assert isinstance(self.test_dataloader,DataLoader)

        self.criterion = criterion # 判别器
        self.optimizer = optimizer # 优化器
        self.scheduler = scheduler # 计数器

        self.best_state = None
        self.last_state = None
        
        self.load_state = None
    
    def training_one_epoch(self,device,dataloader,epoch=0):
        
        assert isinstance(device,torch.device),"DeviceException"
        print("\n--- Total batches {}, 'Training' Epoch : {} ---".format(len(dataloader),epoch))

        self.backbone.train() # train pattern

        train_loss = 0
        correct = 0
        total = 0

        for batch_idx,(inputs,targets) in enumerate(dataloader):

            inputs,targets = inputs.to(device),targets.to(device)
            self.optimizer.zero_grad()
            outputs = self.backbone(inputs)

            loss = self.criterion(outputs,targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 观察Loss是否收敛
            print(
                batch_idx, 
                # len(dataloader),
                'Training... Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    train_loss/(batch_idx+1), 
                    100.*correct/total, 
                    correct, 
                    total
                    )
                )
        
        average_loss = train_loss/len(dataloader)
        average_accuracy = 100.*correct/total
        return average_loss,average_accuracy
    
    def evaluate_one_epoch(self,device,dataloader,epoch=0):

        assert isinstance(device,torch.device),"DeviceException"
        print("\n--- Total batches {}, 'Evaluate' Epoch : {} ---".format(len(dataloader),epoch))

        self.backbone.eval() # evaluate pattern

        evaluate_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx,(inputs,targets) in enumerate(dataloader):
                
                inputs,targets = inputs.to(device),targets.to(device)
                outputs = self.backbone(inputs)

                loss = self.criterion(outputs,targets)
               
                evaluate_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # 观察Loss是否稳定
                print(
                    batch_idx, 
                    # len(dataloader),
                    'Evaluate... Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                        evaluate_loss/(batch_idx+1), 
                        100.*correct/total, 
                        correct, 
                        total
                        )
                    )
        
        average_loss = evaluate_loss/len(dataloader)
        average_accuracy = 100.*correct/total
        return average_loss,average_accuracy
    
    def fit(self,device,epochs,start_epoch=0): # param: controller?
        best_accuracy = 0
        L, A = 0, 0 
        for epoch in range(start_epoch,epochs):
            _, _ = self.training_one_epoch(device,self.train_dataloader,epoch)
            L, A = self.evaluate_one_epoch(device,self.verify_dataloader,epoch) # delete?
            
            if A > best_accuracy:
                self.best_state = {
                    "epoch" : epoch,
                    "param" : self.backbone.state_dict(),
                    "verify" : {'loss' : L, 'accu' : A},
                    "version": "best_state"
                }
                best_accuracy = A
            
            self.scheduler.step()
        
        self.last_state = {
            "epoch" : epochs-1,
            "param" : self.backbone.state_dict(),
            "verify" : {'loss' : L, 'accu' : A},
            "version": "last_state"
        }        

    def test(self,device):
        if self.load_state:
            self.backbone.load_state_dict(self.load_state)
        else: 
            # 若 load_state is None, 则模型将以last_state进行评估
            assert self.last_state,"last_state 不可为空"
        L, A = self.evaluate_one_epoch(device,self.test_dataloader)
        return L, A

    def meta_learn(self,device):
        pass    
 




