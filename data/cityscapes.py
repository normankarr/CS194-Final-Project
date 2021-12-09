import torch
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from torchvision import transforms


src_train_transforms = []
src_train_transforms.append(transforms.Resize(256))
src_train_transforms.append(transforms.PILToTensor())
src_train_transforms.append(transforms.ColorJitter(0.3,0.3,0.3,0.1))
src_train_transforms.append(transforms.ConvertImageDtype(torch.float))
src_train_transforms.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
src_train_transforms = transforms.Compose(src_train_transforms)

src_val_transforms = []
src_val_transforms.append(transforms.Resize(256))
src_val_transforms.append(transforms.PILToTensor())
src_val_transforms.append(transforms.ConvertImageDtype(torch.float))
src_val_transforms.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
src_val_transforms = transforms.Compose(src_val_transforms)

target_transforms = []
target_transforms.append(transforms.Resize(256))
target_transforms.append(transforms.PILToTensor())
target_transforms.append(transforms.Lambda(lambda img: img[0]))
target_transforms.append(transforms.Lambda(lambda img: img.type(torch.long)))
target_transforms = transforms.Compose(target_transforms)

    
def train_dataset(root):
    return Cityscapes(root, split='train', mode='fine', target_type='semantic', transform=src_train_transforms, target_transform=target_transforms)

def val_dataset(root):
    return Cityscapes(root, split='val', mode='fine', target_type='semantic', transform=src_val_transforms, target_transform=target_transforms)

def train_dataloader(root, batch_size, num_workers=0, pin_memory=False):
    train_set = train_dataset(root)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader

def val_dataloader(root, batch_size, num_workers=0, pin_memory=False):
    val_set = val_dataset(root)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    return val_loader