import time
import random
from tqdm import tqdm
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from optim import ScheduledAdam
from model import Model


class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Model().to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)

        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(self.device)

    # def train(self, train_iter=None):
    def train(self, train_iter, val_iter, epochs, length):
        wandb.init(project="my-test-project", entity="nadenny")
        wandb.config = {
            "learning_rate": 0.0001,
            "epochs": 100,
            "batch_size": 30}
        train_len, val_len = length
        best_val_loss = float('inf')
        for epoch in range(epochs):
            print("=============================== Epoch: ", epoch + 1, " of ", epochs,
                  "===============================")
            print('\n')
            batch_loss = 0
            batch_acc = 0
            self.model.train()
            for source, target in tqdm(train_iter):
                source = source.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(source)

                loss = self.criterion(output, target)
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                batch_loss += loss.item()
                batch_acc += self._acc(target, output)

            train_loss = batch_loss / len(train_iter)
            train_acc = batch_acc / train_len

            val_loss, val_acc = self.evaluate(val_iter)
            val_acc = val_acc / val_len

            if val_loss < best_val_loss and epoch != 0:
                self.save_param()

            wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})

            print(f'\rTrain Loss: {train_loss:.3f} | Train Acc.: {train_acc: .3f} |'
                  f'Val. Loss: {val_loss:.3f} | Val. Acc: {val_acc: .3f}'
                  , end='', flush=True)
            print('\n ')

    @torch.no_grad()
    def evaluate(self, valid_iter):
        # self.model.eval()
        batch_loss = 0
        valic_acc = 0
        for source, target in tqdm(valid_iter):
            source = source.to(self.device)
            target = target.to(self.device)

            source = source.to(self.device)
            target = target.to(self.device)

            output = self.model(source)
            loss = self.criterion(output, target)

            batch_loss += loss.item()
            valic_acc += self._acc(target, output)
        valid_loss = batch_loss / len(valid_iter)
        return valid_loss, valic_acc  # valid_acc is un-normalized True Positive

    def inference(self, dataset):
        dataset = dataset.to(self.device)
        self.model.eval()
        with torch.no_grad():
            result = torch.nn.Softmax()(self.model(dataset))
        return result

    def _acc(self, target, output):
        acc = 0
        out = torch.argmax(output, dim=1)
        for i in range(target.shape[0]):
            if target[i] == out[i]:
                acc += 1
        return acc

    def save_param(self):
        torch.save(self.model.state_dict(), 'mymodel.pt')
        torch.save(self.optimizer.state_dict(), 'myoptim.pt')

    def load_param(self):
        self.model.load_state_dict(torch.load('mymodel.pt'))
        self.optimizer.load_state_dict(torch.load('myoptim.pt'))
