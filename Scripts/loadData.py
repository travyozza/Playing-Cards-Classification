import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((200, 200)),
])

card_train = datasets.ImageFolder(root='../../Data/train', transform=transform)
card_val = datasets.ImageFolder(root='../../Data/valid', transform=transform)
card_test = datasets.ImageFolder(root='../../Data/test', transform=transform)

def getTrainDataLoader(batchSize):
    train_dataloader = DataLoader(card_train, batch_size=batchSize, shuffle=True)
    
    return train_dataloader

def getValDataLoader(batchSize):
    validation_dataloader = DataLoader(card_val, batch_size=batchSize, shuffle=False)
    
    return validation_dataloader

def getTestDataLoader(batchSize):
    test_dataloader = DataLoader(card_test, batch_size=batchSize, shuffle=False)
    
    return test_dataloader
    

def dataset_sizes():
    print(f"Training Samples: {len(card_train)} ")
    print(f"Validation Samples: {len(card_val)} ")
    print(f"Test Samples: {len(card_test)} ")