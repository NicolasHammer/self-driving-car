import torch.nn as nn

class PilotNet(nn.Module):
    def __init__(self, num_controls: int, dropout: float = 0.):
        self.convolutional_block = nn.Sequential(
            # 1st layer
            nn.Conv2d(1, 24, kernel_size=(5,5), stride=(2,2)),
            nn.ELU(),
            # 2nd layer
            nn.Conv2d(24, 36, kernel_size=(5,5), stride=(2,2)),
            nn.ELU(),
            # 3rd layer
            nn.Conv2d(36, 48, kernel_size=(5,5), stride=(2,2)),
            nn.ELU(),
            # 4th layer
            nn.Conv2d(48, 64, kernel_size=(3,3)),
            nn.ELU(),
            # 5th layer
            nn.Conv2d(64, 64, kernel_size=(3,3)),
            nn.ELU()
        )

        self.MLP = nn.Sequential(
            # 1st dense layer
            nn.Flatten(),
            nn.Linear(6656, 100),
            nn.ELU(),
            nn.Dropout(dropout),

            # 2nd dense layer
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Dropout(dropout),

            # 3rd dense layer
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        self.output = nn.Linear(10, num_controls)

    def forward(self, image):
        conv_out = self.convolutional_block(image)
        MLP_out = self.MLP(conv_out)
        controls = self.output(MLP_out)
        return controls
