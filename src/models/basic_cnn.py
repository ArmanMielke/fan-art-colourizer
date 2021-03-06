import torch.nn as nn


class BasicCNN(nn.Module):
    model: nn.Sequential

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 8, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, (3, 3), padding=(1, 1)),
            nn.Tanh(),
        )

    def forward(self, image):
        return self.model(image)
