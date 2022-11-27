import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, ".")
from models import ResNet18Classifier, ResNet18ScaledClassifier
from utils import *


SC_RESNET_KD_LR = 0.0005


def training_loop(
    loss_function,
    optimizer,
    student,
    teacher,
    student_save_path,
    epochs,
    train_loader,
    val_loader,
    test_loader,
):
    if torch.cuda.is_available():
        teacher.cuda()
    best_accuracy = 0.0

    for epoch in range(epochs):
        student.train(True)
        if torch.cuda.is_available():
            student = student.cuda()

        current_loss = 0.0
        current_accuracy = 0.0
        total_count = 0.0

        for data in tqdm(train_loader, desc=f"Epoch: {epoch + 1}"):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            with torch.no_grad():
                teacher_logits = teacher(inputs)

            optimizer.zero_grad()
            outputs = student(inputs)
            loss = loss_function(outputs, teacher_logits)
            loss.backward()
            optimizer.step()

            current_loss += loss.item() * inputs.size(0)
            out = torch.argmax(outputs.detach(), dim=1)
            current_accuracy += (labels == out).sum().item()
            total_count += len(labels)

        print(
            f"Train loss: {current_loss/len(train_loader)}, Train Acc: {current_accuracy*100/total_count}%"
        )

        current_accuracy = evaluate(student, val_loader)
        print(f"Val accuracy:{current_accuracy*100}%")
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            checkpoint(student, student_save_path)

    print(f"Test Accuracy: {evaluate(student, test_loader) * 100}%")


if __name__ == "__main__":
    _script, dataset, epochs = sys.argv
    epochs = int(epochs)

    if dataset == "CIFAR10":
        train_loader, val_loader, test_loader = prepare_cifar10_dataset(
            CIFAR10_TRAIN_VAL_SPLIT, CIFAR10_BATCH_SIZE
        )
        num_classes = CIFAR10_NUM_CLASSES
        student_model_save_path = "{}{}_kd_{}".format(
            MODEL_SAVE_PATH_DIR,
            CIFAR10_MODEL_SAVE_PATH_PREFIX,
            RESNET18_SC_MODEL_SAVE_PATH_SUFFIX,
        )
        teacher_model_save_path = "{}{}{}".format(
            MODEL_SAVE_PATH_DIR,
            CIFAR10_MODEL_SAVE_PATH_PREFIX,
            RESNET18_MODEL_SAVE_PATH_SUFFIX,
        )
    else:
        raise Exception("Unsupported dataset. Allowed: CIFAR10")

    student = ResNet18ScaledClassifier(num_classes)
    teacher = ResNet18Classifier(num_classes)
    teacher.load_state_dict(torch.load(teacher_model_save_path))

    optimizer = torch.optim.Adam(
        student.parameters(), lr=SC_RESNET_KD_LR, weight_decay=1e-5
    )
    training_loop(
        torch.nn.MSELoss(),
        optimizer,
        student,
        teacher,
        student_model_save_path,
        epochs,
        train_loader,
        val_loader,
        test_loader,
    )
