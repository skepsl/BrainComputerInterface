import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class Conv2D(nn.Module):
    def __init__(self):
        super(Conv2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.BatchNorm2d(num_features=32),
            nn.ELU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.BatchNorm2d(num_features=64),
            nn.ELU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.BatchNorm2d(num_features=128),
            nn.ELU())

    def forward(self, x):
        x = self.conv(x)
        return x


class TimeDistributedLayer(nn.Module):
    def __init__(self, module):
        super(TimeDistributedLayer, self).__init__()
        self.module = module
        self.ff = nn.Linear(in_features=14080, out_features=1024)

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = Rearrange('b s c h w -> (b s) c h w', b=b, s=s)(x)
        x = self.module(x)
        x = Rearrange('(b s) c h w-> (b s) (c h w)', b=b, s=s, c=128, h=h, w=w)(x)
        x = nn.Dropout(0.5)(nn.ELU()(self.ff(x)))
        x = Rearrange('(b s) d -> b s d', b=b, s=s)(x)
        return x


class LSTM(nn.Module):
    def __init__(self, ):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, batch_first=True, dropout=0.5)

    def forward(self, x):
        output, (_, _) = self.lstm(x)
        output = output[:, -1]
        return output


class SoTA(nn.Module):
    def __init__(self):
        super(SoTA, self).__init__()
        self.ff = nn.Linear(in_features=1024, out_features=1024)
        self.out = nn.Linear(in_features=1024, out_features=5)

    def forward(self, x):
        x = nn.ELU()(self.ff(x))
        x = self.out(x)
        return x


if __name__ == '__main__':
    inputs = torch.zeros((300, 10, 1, 10, 11))

    convlayer = Conv2D()

    conv = TimeDistributedLayer(convlayer)
    lstm = LSTM()
    sota = SoTA()

    a = conv(inputs)
    b = lstm(a)
    c = sota(b)

    print(c.shape)
