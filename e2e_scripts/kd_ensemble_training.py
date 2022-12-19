from copy import deepcopy
import sys

import torch.nn as nn
import torch.nn.functional as F
from torchensemble import BaggingClassifier
from torchensemble.utils import io
from tqdm import tqdm

sys.path.insert(0, ".")
from models import ResNet18ScaledClassifier
from utils import *


KD_RESNET_ENSEMBLE_LR = 0.0005


class CustomLogger:
    def __init__(self):
        pass

    def info(self, message):
        print(message)


def geom_avg_kld_loss(student_output, teacher_outputs):
    return sum(
        [
            nn.KLDivLoss()(
                F.log_softmax(student_output, dim=1),
                F.softmax(teacher_output, dim=1),
            )
            for teacher_output in teacher_outputs
        ]
    ) / len(teacher_outputs)


def geom_avg_mse_loss(student_output, teacher_outputs):
    return sum(
        [
            nn.MSELoss()(
                student_output,
                teacher_output,
            )
            for teacher_output in teacher_outputs
        ]
    ) / len(teacher_outputs)


def training_loop(
    loss_function,
    ensemble,
    epochs_per_student,
    save_path,
    train_loader,
    val_loader,
    test_loader,
):
    logger = CustomLogger()

    for estimator_idx, estimator in enumerate(ensemble.estimators_):
        ensemble.train(False)

        best_accuracy = 0.0
        best_model = None

        current_model = deepcopy(estimator)
        optimizer = torch.optim.Adam(
            current_model.parameters(), lr=KD_RESNET_ENSEMBLE_LR, weight_decay=1e-5
        )

        for epoch in range(epochs_per_student):
            current_model.train(True)
            total_count = 0
            correct_count = 0
            current_loss = 0.0

            for data in tqdm(
                train_loader, desc=f"Epoch: {epoch + 1}, Estimator: {estimator_idx + 1}"
            ):
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()

                with torch.no_grad():
                    teacher_outputs = [
                        teacher(inputs) for teacher in ensemble.estimators_
                    ]

                optimizer.zero_grad()
                student_outputs = current_model(inputs)
                loss = loss_function(student_outputs, teacher_outputs)
                loss.backward()
                optimizer.step()

                current_loss += loss.item()
                out = torch.argmax(student_outputs.detach(), dim=1)
                correct_count += (labels == out).sum().item()
                total_count += len(labels)

            print(
                f"Estimator:{estimator_idx + 1}, Train loss: {current_loss/len(train_loader)}, Train Acc: {correct_count*100/total_count}%"
            )

            current_accuracy = evaluate(current_model, val_loader)
            print(f"Val accuracy:{current_accuracy*100}%")
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_model = deepcopy(current_model)

        test_accuracy = evaluate(best_model, test_loader)
        print(f"Estimator: {estimator_idx + 1}, Test Accuracy: {test_accuracy}")
        ensemble.estimators_[estimator_idx] = best_model
        io.save(ensemble, save_path, logger)


if __name__ == "__main__":
    _script, dataset, epochs = sys.argv
    epochs = int(epochs)

    if dataset == "CIFAR10":
        train_loader, val_loader, test_loader = prepare_cifar10_dataset(
            CIFAR10_TRAIN_VAL_SPLIT, CIFAR10_BATCH_SIZE
        )
        num_classes = CIFAR10_NUM_CLASSES
        model_save_path_prefix = (
            f"{MODEL_SAVE_PATH_DIR}{CIFAR10_MODEL_SAVE_PATH_PREFIX}"
        )
    elif dataset == "EMNIST":
        train_loader, val_loader, test_loader = prepare_emnist_dataset(
            EMNIST_TRAIN_VAL_SPLIT, EMNIST_BATCH_SIZE
        )
        num_classes = EMNIST_NUM_CLASSES
        model_save_path_prefix = f"{MODEL_SAVE_PATH_DIR}{EMNIST_MODEL_SAVE_PATH_PREFIX}"
    else:
        raise Exception("Unsupported dataset. Allowed: CIFAR10, EMNIST")
    ensemble_save_path = f"{model_save_path_prefix}ensemble_direct"
    ensemble = BaggingClassifier(
        estimator=ResNet18ScaledClassifier,
        n_estimators=ENSEMBLE_SIZE,
        estimator_args={"num_classes": num_classes},
        cuda=torch.cuda.is_available(),
    )
    io.load(ensemble, ensemble_save_path)
    kd_ensemble_save_path = f"{model_save_path_prefix}ensemble_kd"
    training_loop(
        geom_avg_mse_loss,
        ensemble,
        epochs,
        kd_ensemble_save_path,
        train_loader,
        val_loader,
        test_loader,
    )
