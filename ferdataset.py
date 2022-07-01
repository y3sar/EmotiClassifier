from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor;
import torch;


DATASET_ROOT = './dataset/'
BATCH_SIZE = 64;
NUM_CLASS = 7;

transforms = {x : ToTensor() for x in ['train', 'test']}

def target_to_oh(target):
    one_hot = torch.eye(NUM_CLASS)[target]
    return one_hot


datasets = {x : ImageFolder(DATASET_ROOT + x, transform=transforms[x], target_transform=target_to_oh) for x in ['train', 'test']}

dataloaders = {x : (DataLoader(datasets[x], batch_size=BATCH_SIZE, shuffle=True) if x == 'train' else DataLoader(datasets[x], batch_size=BATCH_SIZE)) for x in ['train', 'test']}


print(datasets['train'][0][1]);

