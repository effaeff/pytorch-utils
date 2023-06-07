import numpy as np


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_acc = -np.inf

    def early_stop(self, acc):
        if acc > self.max_acc:
            self.max_acc = acc
            self.counter = 0
        elif acc < (self.max_acc + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
