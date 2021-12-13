import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader

def get_train_dataloader(dataset_class):
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
        ]
    )

    if issubclass(dataset_class,Dataset):
        dataset = dataset_class(
            root='./datasets/data',
            train=True, 
            download=True,
            transform=transform
            )
    else:
        raise Exception("The input class is not a subclass of torch.utils.data.dataset.Dataset")
    
    dataloader = DataLoader(dataset,batch_size=128,shuffle=True,num_workers=2)
    return dataloader

def get_test_dataloader(dataset_class):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
        ]
    )
    if issubclass(dataset_class,Dataset):
        dataset = dataset_class(
            root = './datasets/data',
            train=False,
            download=True,
            transform=transform
        )
    else:
        raise Exception("The input class is not a subclass of torch.utils.data.dataset.Dataset")
    
    dataloader = DataLoader(dataset,batch_size=100,shuffle=False,num_workers=2)
    return dataloader   

if __name__=="__main__":
    loader = get_train_dataloader(tv.datasets.CIFAR10)