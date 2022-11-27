import sys

from tqdm import tqdm

sys.path.insert(0, ".")
from models import ResNet18Classifier, ResNet18ScaledClassifier
import torch.nn.functional as F
from utils import *


RESNET_BASE_LR = 0.0001
RESNET_FC_LR = 0.001

SC_RESNET_LR = 0.0005


def training_loop(
    loss_function,
    optimizer,
    model,
    model_save_path,
    epochs,
    train_loader,
    val_loader,
    test_loader,
):
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train(True)
        if torch.cuda.is_available():
            model = model.cuda()

        current_loss = 0.0
        current_accuracy = 0.0
        total_count = 0.0

        for data in tqdm(train_loader, desc=f"Epoch: {epoch + 1}"):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = F.softmax(model(inputs), dim=1)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            current_loss += loss.item() * inputs.size(0)
            out = torch.argmax(outputs.detach(), dim=1)
            current_accuracy += (labels == out).sum().item()
            total_count += len(labels)

        print(
            f"Train loss: {current_loss/len(train_loader)}, Train Acc: {current_accuracy*100/total_count}%"
        )
        current_accuracy = evaluate(model, val_loader)
        print(f"Val accuracy:{current_accuracy*100}%")
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            checkpoint(model, model_save_path)

    print(f"Test Accuracy: {evaluate(model, test_loader) * 100}%")


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
                {"params": model.base.parameters(), "lr": RESNET_BASE_LR},
                {"params": model.classification_layer.parameters(), "lr": RESNET_FC_LR},
            ],
            weight_decay=1e-5,
        )
        model_save_path = f"{model_save_path_prefix}{RESNET18_MODEL_SAVE_PATH_SUFFIX}"
    elif model == "resnet18_sc":
        model = ResNet18ScaledClassifier(num_classes)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=SC_RESNET_LR, weight_decay=1e-5
        )
        model_save_path = (
            f"{model_save_path_prefix}{RESNET18_SC_MODEL_SAVE_PATH_SUFFIX}"
        )
    else:
        raise Exception("Unsupported model: resnet18 allowed")

    loss_function = torch.nn.CrossEntropyLoss()
    training_loop(
        loss_function,
        optimizer,
        model,
        model_save_path,
        epochs,
        train_loader,
        val_loader,
        test_loader,
    )
