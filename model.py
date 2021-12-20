import torch
import torch.nn as nn
from model_dict import Conv2D, TimeDistributedLayer, LSTM, SoTA


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = Conv2D()
        self.sequence = TimeDistributedLayer(self.conv)
        self.lstm = LSTM()
        self.sota = SoTA()

    def forward(self, x):
        x = self.sequence(x)
        x = self.lstm(x)
        x = self.sota(x)
        return x

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = Model()
    inputs = torch.zeros((300, 10, 1, 10, 11))

    test = model(inputs)
    print(f'Params: {model.count_params()}')
    print(test.shape)