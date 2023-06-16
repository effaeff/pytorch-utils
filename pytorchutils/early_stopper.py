import numpy as np


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, max=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = -np.inf if max else np.inf
        self.max = max

    def early_stop(self, acc):
        if self.max:
            if acc > self.best:
                self.best = acc
                self.counter = 0
            elif acc < (self.best + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        else:
            if acc < self.best:
                self.counter = 0
            elif acc > (self.best + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        return False
