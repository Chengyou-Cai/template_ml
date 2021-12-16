import os
import torch

from .container import ModelContainer

class Controller(object):

    def __init__(self,model_container=None,view_container=None):
        assert isinstance(model_container,ModelContainer),"TypeException in mvc.py line39"
        
        self.model = model_container
        self.view = view_container

        self.configs = None
        self.device = None

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
        model_ckpt = torch.load(load_ckpt_path)

        self.model.backbone.load_state_dict(model_ckpt['state_'])
        

    def save_checkpoint(self,save_ckpt_path=None):
        if not save_ckpt_path:
            save_ckpt_path = self.ckpt_path
        
        ckpt_dir = os.path.dirname(save_ckpt_path)
        if not os.path.isdir(ckpt_dir):
            os.mkdir(ckpt_dir)
        
        accuracy = self.model.test(self.device)
        print("Accuracy:{}%   Saving checkpoint ...".format(accuracy))
        model_ckpt = {
            'state_': self.model.backbone.state_dict(),
            'metric': "Accuracy:{}".format(accuracy)
        }
        torch.save(model_ckpt,save_ckpt_path)

    def init_model(self):
        if not self.configs:
            self.set_configs()
        self.model.backbone.to(self.device)
        
        # if self.configs['device'] != "cpu":
        #     self.model.backbone = nn.DataParallel(self.model.backbone)
        #     torch.backends.cudnn.benchmark = True
    
    def train_model(self):
        self.model.fit(self.device,self.configs['epochs'])
        self.save_checkpoint()
    
    def test_model(self):
        self.model.test(self.device)