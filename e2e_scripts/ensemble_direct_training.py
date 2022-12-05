import sys

from torchensemble import BaggingClassifier

sys.path.insert(0, ".")
from models import ResNet18ScaledClassifier
from utils import *


SC_RESNET_ENSEMBLE_LR = 0.0005


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

    model_save_path = f"{model_save_path_prefix}ensemble_direct"
    ensemble = BaggingClassifier(
        estimator=ResNet18ScaledClassifier,
        n_estimators=ENSEMBLE_SIZE,
        estimator_args={"num_classes": num_classes},
        cuda=torch.cuda.is_available(),
    )
    ensemble.set_criterion(torch.nn.CrossEntropyLoss())
    ensemble.set_optimizer(
        optimizer_name="Adam", lr=SC_RESNET_ENSEMBLE_LR, weight_decay=1e-5
    )
    ensemble.fit(
        train_loader=train_loader,
        epochs=epochs,
        test_loader=val_loader,
        save_dir=model_save_path,
    )
    test_accuracy = ensemble.evaluate(test_loader=test_loader)
    print(f"Test Accuracy: {test_accuracy * 100}%")
