from torch import nn
from torch.utils.data import Dataset,DataLoader

class ModelContainer(object):

    def __init__(
        self,
        model_name='default_backbone_dataset',
        backbone=None,dataset_=None,
        train_dataloader_getfunc=None,
        verify_dataloader_getfunc=None,
        test_dataloader_getfunc=None,
        criterion=None,
        optimizer=None,
        scheduler=None,
        ):
        assert isinstance(backbone,nn.Module),"backbone isn't an instance of torch.nn.Module"
        assert issubclass(dataset_,Dataset),"dataset_ isn't a subclass of torch.utils.data.Dataset"
        
        self.model_name = model_name
        self.backbone = backbone
        self.dataset_ = dataset_
        
        self.train_dataloader = train_dataloader_getfunc(self.dataset_) if train_dataloader_getfunc else None
        self.verify_dataloader = verify_dataloader_getfunc(self.dataset_) if verify_dataloader_getfunc else None
        self.test_dataloader = test_dataloader_getfunc(self.dataset_) if test_dataloader_getfunc else None

        if self.train_dataloader:
            assert isinstance(self.train_dataloader,DataLoader)
        if self.verify_dataloader:
            assert isinstance(self.verify_dataloader,DataLoader)
        if self.test_dataloader:
            assert isinstance(self.test_dataloader,DataLoader)

        self.criterion = criterion # 判别器
        self.optimizer = optimizer # 优化器
        self.scheduler = scheduler # 计数器