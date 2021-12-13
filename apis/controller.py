import os
import torch

from .train import ModelContainer,TrainUnit


class Controller(object):

    def __init__(self,model=None,view=None):
        assert isinstance(model,ModelContainer),"TypeException in mvc.py line39"
        
        self.model = model
        self.view = view

        self.configs = None
        self.device = None

        self.best_acc = 0
        self.start_epoch = 0

        self.ckpt_path = r'./models/{}/best_ckpt.pth'.format(self.model.model_name)

    def set_configs(self,configs=None,epochs=15,gpu_id=0):
        if configs:
            self.configs = configs
        else:
            self.configs = dict()
        
        self.configs['epochs'] = epochs
        self.configs['gpu_id'] = gpu_id
        self.configs['device'] = "cuda:{}".format(self.configs['gpu_id']) if torch.cuda.is_available() else "cpu" 

        self.device = torch.device(self.configs['device'])
        
        print(self.configs)
    
    def load_checkpoint(self,load_ckpt_path=None):
        if not load_ckpt_path:
            load_ckpt_path = self.ckpt_path
        assert os.path.exists(load_ckpt_path),"ckpt_path doesn't exist"
        checkpoint = torch.load(load_ckpt_path)
        self.model.backbone.load_state_dict(checkpoint['model'])
        
        self.best_acc = checkpoint['acc']
        self.start_epoch = checkpoint['epoch']

    def init_model(self):
        if not self.configs:
            self.set_configs()
        self.model.backbone.to(self.device)
        
        # if self.configs['device'] != "cpu":
        #     self.model.backbone = nn.DataParallel(self.model.backbone)
        #     torch.backends.cudnn.benchmark = True
    
    def train_model(self,save_ckpt_path=None):
        if not save_ckpt_path:
            save_ckpt_path = self.ckpt_path
        train_unit =TrainUnit(
            self.model,
            self.device,
            self.best_acc,
            self.start_epoch,
            save_ckpt_path
            )
        train_unit.train_for_epochs(self.configs['epochs'])
    
    def test_model(self):
        pass