import torch.nn as nn

class BBoxHead(nn.Module):
    def __init__(self, num_classes):
        super(BBoxHead, self).__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(7*7*512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        self.cls_fc = nn.Linear(4096, num_classes)
        self.reg_fc = nn.Linear(4096, num_classes*4)

    def forward(self, features):
        x = self.shared_layers(features)
        cls_scores = self.cls_fc(x)
        reg_scores = self.reg_fc(x)

        return cls_scores, reg_scores

