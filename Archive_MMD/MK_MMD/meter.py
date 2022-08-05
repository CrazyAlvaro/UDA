from typing import Optional, List

# https://github.com/thuml/Transfer-Learning-Library/blob/7b0ccb3a8087ecc65daf4b1e815e5a3f42106641/common/utils/meter.py

class AverageMeter(object):
    def __init__(self, name: str, fmt: Optional[str] = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} {avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)