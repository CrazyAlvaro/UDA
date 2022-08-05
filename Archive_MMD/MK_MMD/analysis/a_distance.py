from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import sgd
from meter import AverageMeter
from metric import binary_accuracy

# https://tl.thuml.ai/common/utils/analysis.html
# https://github.com/thuml/Transfer-Learning-Library/blob/7b0ccb3a8087ecc65daf4b1e815e5a3f42106641/common/utils/analysis/a_distance.py#L1

class ANet(nn.Module):
    def __init__(self, in_feature):
        super(ANet, self).__init__()
        self.layer = nn.Linear(in_feature, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x

def calculate(source_feature: torch.Tensor, target_feature: torch.Tensor,
             device, progress=True, training_expochs=10):
    """
    Calculate the A-distance, which is a measure for distribution discrepancy.

    The definition is dist = 2 * (1 - 2*epsilon), where epsilon is the test error of 
    a classifier trained to discriminate the source from the target
    """
    source_label = torch.ones((source_feature.shape[0], 1))
    target_label = torch.zeros((target_feature.shape[0], 1))

    feature = torch.cat([source_feature, target_feature], sim=0)
    label = torch.cat([source_label, target_label], dim=0)

    datasest = TensorDataset(feature, label)
    length = len(datasest)
    train_size = int(0.8 * length)
    val_size = length - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    anet = ANet(feature.shape[1]).to(device)
    optimizer = sgd(anet.parameters(), lr=0.01)

    a_distance = 2.0
    for epoch in range(training_expochs):
        anet.train()
        for (x, label) in train_loader:
            x = x.to(device)
            label = label.to(device)

            anet.zero_grad()
            y = anet(x)
            loss = F.binary_cross_entropy(y, label)
            loss.backward()
            optimizer.step()
        
        anet.eval()
        meter = AverageMeter("accuracy", ":4.2f")
        with torch.no_grad():
            for (x, label) in val_loader:
                x = x.to(device)
                label = label.to(device)
                y = anet(x)
                acc = binary_accuracy(y, label)
                meter.update(acc, x.shape[0])
        
        error = 1 - meter.avg / 100
        a_distance = 2 * (1 - 2 * error)
        
        if progress:
            print("epoch {} accuracy: {} A-dist: {}".format(epoch, meter.avg, a_distance))