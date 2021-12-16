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

        self.ckpt_dir = r'./models/{}/'.format(self.model.model_name)

    def set_configs(self,configs=None,epochs=1,gpu_id=0):
        if configs:
            self.configs = configs
        else:
            self.configs = dict()
        
        self.configs['epochs'] = epochs
        self.configs['gpu_id'] = gpu_id
        self.configs['device'] = "cuda:{}".format(self.configs['gpu_id']) if torch.cuda.is_available() else "cpu" 

        self.device = torch.device(self.configs['device'])
        
        print(self.configs)
    
    def load_checkpoint(self,load_path=None):
        if not load_path:
            load_path = self.ckpt_dir+'save_state.pth'
        assert os.path.exists(load_path),"ckpt_path doesn't exist"
        
        print("--- Load {} ---".format(load_path))
        state_ckpt = torch.load(load_path)

        return state_ckpt

    def save_checkpoint(self,state_ckpt,save_path=None):
        if not save_path:
            save_path = self.ckpt_dir+'save_state.pth'
        
        if not os.path.isdir(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
        
        print("--- Save {} ---".format(save_path))
        torch.save(state_ckpt,save_path)

    def init_model(self):
        if not self.configs:
            self.set_configs()
        self.model.backbone.to(self.device)
        
        # if self.configs['device'] != "cpu":
        #     self.model.backbone = nn.DataParallel(self.model.backbone)
        #     torch.backends.cudnn.benchmark = True
    
    def train_model(self):
        self.model.fit(self.device,self.configs['epochs'])
        self.save_checkpoint(state_ckpt=self.model.best_state,save_path=self.ckpt_dir+'best_state.pth')
        self.save_checkpoint(state_ckpt=self.model.last_state,save_path=self.ckpt_dir+'last_state.pth')
    
    def test_model(self):
        self.model.load_state = self.load_checkpoint(load_path=self.ckpt_dir+'best_state.pth')
        self.model.test(self.device)