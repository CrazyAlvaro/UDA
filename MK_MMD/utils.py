import torch
import torch.nn as nn
import torch.nn.functional as F

from meter import AverageMeter

def validate(val_loader, model, args, device) -> float:
    """
    """
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, _ = accuracy(output, target, topk=(1,))

            top1.update(acc1.item(), images.size(0))

    return top1.avg