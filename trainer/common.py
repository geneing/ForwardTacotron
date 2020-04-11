from torch.utils.data.dataloader import DataLoader


class Session:

    def __init__(self,
                 index: int,
                 r: int,
                 lr: int,
                 max_step: int,
                 bs: int,
                 train_set: DataLoader,
                 val_set: DataLoader) -> None:
        self.index = index
        self.r = r
        self.lr = lr
        self.max_step = max_step
        self.bs = bs
        self.train_set = train_set
        self.val_set = val_set
        self.val_sample = next(iter(val_set))


class Averager:

    def __init__(self):
        self.count = 0
        self.val = 0.

    def add(self, val):
        self.val += float(val)
        self.count += 1

    def reset(self):
        self.val = 0.
        self.count = 0

    def get(self):
        return self.val / self.count

