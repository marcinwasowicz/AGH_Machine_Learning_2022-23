from copy import deepcopy
import sys

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

sys.path.insert(0, ".")
from models import ResNet18Classifier


CIFAR10_TRAIN_VAL_SPLIT = [42000, 8000]
CIFAR10_BATCH_SIZE = 64
CIFAR10_MODEL_SAVE_PATH_PREFIX = "cifar10_direct_"
CIFAR10_NUM_CLASSES = 10

RESNET18_MODEL_SAVE_PATH_SUFFIX = "resnet18.pth"

MODEL_SAVE_PATH_DIR = "./model_binaries/"


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


def checkpoint(model, model_save_path):
    model_state_dict = deepcopy(model.cpu().state_dict())
    torch.save(model_state_dict, model_save_path)


def prepare_cifar10_dataset(train_val_split, batch_size):
    train_data = CIFAR10(root="./Dataset", train=True, transform=transform(224))
    train_data, val_data = random_split(
        train_data, train_val_split, generator=torch.Generator().manual_seed(42)
    )
    test_data = CIFAR10(root="./Dataset", train=False, transform=transform(224))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    _script, model, dataset, epochs = sys.argv
    epochs = int(epochs)

    if dataset == "CIFAR10":
        train_loader, val_loader, test_loader = prepare_cifar10_dataset(
            CIFAR10_TRAIN_VAL_SPLIT, CIFAR10_BATCH_SIZE
        )
        num_classes = CIFAR10_NUM_CLASSES
        model_save_path_prefix = (
            f"{MODEL_SAVE_PATH_DIR}{CIFAR10_MODEL_SAVE_PATH_PREFIX}"
        )
    else:
        raise Exception("Unsupported dataset. Allowed: CIFAR10")

    if model == "resnet18":
        model = ResNet18Classifier(num_classes)
        optimizer = torch.optim.Adam(
            [
                {"params": model.base.parameters(), "lr": 0.0001},
                {"params": model.classification_layer.parameters(), "lr": 0.001},
            ]
        )
        model_save_path = f"{model_save_path_prefix}{RESNET18_MODEL_SAVE_PATH_SUFFIX}"
    else:
        raise Exception("Unsupported model: resnet18 allowed")

    best_accuracy = 0.0
    loss_function = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()

    for epoch in range(epochs):
        model.train(True)

        current_loss = 0.0
        current_accuracy = 0.0

        for data in tqdm(train_loader, desc=f"Epoch: {epoch + 1}"):
            inputs, labels = data

            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            current_loss += loss.item() * inputs.size(0)
            out = torch.argmax(outputs.detach(), dim=1)
            assert out.shape == labels.shape
            current_accuracy += (labels == out).sum().item()

        print(
            f"Train loss: {current_loss/len(train_loader)}, Train Acc: {current_accuracy*100/len(train_loader)}%"
        )

        model.train(False)
        current_accuracy = evaluate(model, val_loader)
        print(f"Val accuracy:{current_accuracy*100}%")
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            checkpoint(model, model_save_path)
