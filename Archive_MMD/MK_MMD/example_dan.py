# It's not serve as running code but document
# https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/dan.py

import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dan import MultipleKernelMaximumMeanDiscrepancy, ImageClassifier
from metric import accuracy 
from meter import AverageMeter, ProgressMeter
from kernels import GaussianKernel
from logger import CompleteLogger 
from analysis import collect_feature, tsne, a_distance
import utils

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)

    train_transform = None
    val_transform   = None

    train_source_loader = DataLoader(train_source_dataset)
    train_target_loader = DataLoader(train_target_dataset)

    val_loader  = DataLoader(val_dataset)
    test_loader = DataLoader(test_dataset)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    classifier = ImageClassifier(backbone, num_classes, bottlenect_dim, pool_layer, finetune).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters())
    lr_scheduler = LambdaLR(optimizer)

    ##################################################
    # define loss function
    #   Add multiple kernels, later will use to compute MK_MMD loss for fully connected features
    #   from both source and target input
    ##################################################
    mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        kernels=[GaussianKernel(alpha= 2 ** k) for k in range(-3,2)],
        linear = True,
    )

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)

        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)

        A_distance = a_distance.calculate(source_feature, target_feature, device)

        return
    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return
    
    # start training
    best_acc1 = 0.
    
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, mkmmd_loss, optimizer,
              lr_scheduler, epoch, args)
        
        # evalute on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)

        # remember the best acc1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evalute on the test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()

def train(train_source_iter: ForeverDataIterator, train_target_iter: FoeverDataIterator, model: ImageClassifier,
          mkmmd_loss: MultipleKernelMaximumMeanDiscrepancy, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):

    progress = ProgressMeter()

    # switch to train mode
    model.train()
    mkmmd_loss.train()

    for _ in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t, labels_t = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)

        # model prediction y and fully connect output f
        y_s, f_s = model(x_s)
        y_t, f_t = model(x_t)

        ##################################################
        # compute loss:
        #   loss = classification_loss + transfer_loss
        #   transfer_loss is  "multi kernel maximum mean discrepancy loss"
        #
        #   idea: is to minimize both classification loss, and difference between
        #         fully connected features extracted from both source domain and target domain
        ##################################################
        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = mkmmd_loss(f_s, f_t)
        loss = cls_loss + transfer_loss * args.trade_off

        # cls_acc = accuracy(y_s, labels_s)[0]
        # tgt_acc = accuracy(y_t, labels_t)[0]

        # compute gradient and SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()