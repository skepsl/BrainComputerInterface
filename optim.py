import numpy as np
import torch


class ScheduledAdam():
    def __init__(self, optimizer, hidden_dim, warm_steps):
        self.init_lr = np.power(hidden_dim, -0.5)
        self.optimizer = optimizer
        self.current_steps = 0
        self.warm_steps = warm_steps

    def step(self):
        # Update learning rate using current step information
        self.current_steps += 1
        lr = self.init_lr * self.get_scale()

        for p in self.optimizer.param_groups:
            p['lr'] = lr

        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_scale(self):
        return np.min([
            np.power(self.current_steps, -0.5),
            self.current_steps * np.power(self.warm_steps, -0.5)
        ])

    def save_param(self):
        torch.save(self.optimizer.state_dict(), 'optim.pt')

    def load_param(self):
        self.optimizer.load_state_dict(torch.load('optim.pt'))