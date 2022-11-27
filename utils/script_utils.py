from copy import deepcopy

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10


CIFAR10_TRAIN_VAL_SPLIT = [42000, 8000]
CIFAR10_BATCH_SIZE = 64
CIFAR10_MODEL_SAVE_PATH_PREFIX = "cifar10_"
CIFAR10_NUM_CLASSES = 10

RESNET18_SC_MODEL_SAVE_PATH_SUFFIX = "resnet18_sc.pth"
RESNET18_MODEL_SAVE_PATH_SUFFIX = "resnet18.pth"

MODEL_SAVE_PATH_DIR = "./model_binaries/"


def checkpoint(model, model_save_path):
    model_state_dict = deepcopy(model.cpu().state_dict())
    torch.save(model_state_dict, model_save_path)


def transform(resize_shape):
    return transforms.Compose(
        [
            transforms.Resize(resize_shape),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def evaluate(model, loader):
    correct = 0
    total = 0
    model.train(False)

    if torch.cuda.is_available():
        model = model.cuda()

    with torch.no_grad():
        for inputs, labels in loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            out = model(inputs).cpu()
            out = torch.argmax(out, dim=1)
            acc = (out == labels).sum().item()
            total += len(labels)
            correct += acc
    return correct / total


def prepare_cifar10_dataset(train_val_split, batch_size):
    train_data = CIFAR10(
        root="./Dataset", download=True, train=True, transform=transform(224)
    )
    train_data, val_data = random_split(
        train_data, train_val_split, generator=torch.Generator().manual_seed(42)
    )
    test_data = CIFAR10(
        root="./Dataset", download=True, train=False, transform=transform(224)
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
