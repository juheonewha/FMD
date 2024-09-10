import torch
import torch.nn as nn
import torchvision.models as models

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(in_features=2048, out_features=128, bias=True)
        self.fc_end = nn.Linear(128, 1)

    def forward_once(self, x):
        output = self.model(x)
        output = torch.sigmoid(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        difference = torch.abs(output1 - output2)
        output = self.fc_end(difference)
        return output