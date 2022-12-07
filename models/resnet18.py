import torch.nn as nn
from torchvision.models.resnet import resnet18


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = resnet18(pretrained=True)
        self.dropout = nn.Dropout()
        self.classification_layer = nn.Linear(
            list(self.base.children())[-1].out_features, num_classes
        )

    def forward(self, x):
        x = self.base(x)
        x = x.view(-1, self.classification_layer.in_features)
        x = self.dropout(x)
        return self.classification_layer(x)
