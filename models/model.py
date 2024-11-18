import torch
from torch import nn

from models.base import BaseModel


class DummyModel(BaseModel):

    def __init__(self):
        # type: () -> None

        super().__init__()
        self.net = nn.Sequential(
            # ->> down-sample block
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.SiLU(inplace=True),
            # ->> up-sample block
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(inplace=True),
            # ->> final conv
            nn.Conv2d(32, 3, kernel_size=1)
        )


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        return self.net(x)


# ---------

def main():
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DummyModel().to(device)

    x = torch.rand((batch_size, 3, 128, 128)).to(device)
    y = model.forward(x)

    print(f'$> input shape: {tuple(x.shape)}')
    print(f'$> output shape: {tuple(y.shape)}')


if __name__ == '__main__':
    main()
