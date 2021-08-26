from brainmri.optimizer.optimize import *

import argparse
from datetime import date

import torch
import torch.nn as nn
import torch.optim as optim

import segmentation_models_pytorch as smp


class TrainEpochWithScheduler(smp.utils.train.TrainEpoch):
    def __init__(self, model, loss, metrics, optimizer, scheduler, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=device,
            verbose=verbose,
        )
        self.scheduler = scheduler

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss, prediction


def train_model(config, train_dl, valid_dl):
    train_loss, valid_loss, train_fscore, valid_fscore, train_iou, valid_iou = [], [], [], [], [], []
    model_type = config.get("model", "unet")
    encoder = config.get("model_encoder", "resnet50")
    encoder_weights = config.get("encoder_weights", "imagenet")
    activation = config.get("model_act", "sigmoid")
    classes = config.get("num_classes", 1)
    lr = config.get("lr", 0.0001)
    epochs = config.get("epochs", 10)
    lr_scheduler_name = config.get("lr_scheduler", None)
    optimizer_name = config.get("optimizer", "adam")
    device = config.get("device", "cpu")
    
    model = None
    if model_type == "unet":
        model = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                         classes=classes, activation=activation)
    elif model_type == "fpn":
        model = smp.FPN(encoder_name=encoder, encoder_weights=encoder_weights,
                         classes=classes, activation=activation)
    else:
        print("Model type is not valid. Training cannot be done...")
        return
    
    print("Training", model_type, "w/", encoder, "backbone")
    loss = smp.utils.losses.DiceLoss()
    metrics = [smp.utils.metrics.IoU(threshold=0.5), smp.utils.metrics.Fscore(threshold=0.5)]
    
    optimizer = None
    
    if optimizer_name == "adam":
        optimizer = adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = sgd(model.parameters(), lr=lr)
        
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr/10, steps_per_epoch=len(train_dl), epochs=epochs)
    scheduler = None    
    if torch.cuda.device_count() > 1 and device == "cuda":
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        model = model.cuda()
    
    train_epoch = None
    
    if scheduler:
        train_epoch = TrainEpochWithScheduler(model, loss=loss, metrics=metrics, 
                                              optimizer=optimizer, scheduler=scheduler, 
                                              device=device, verbose=True)
    else:
        train_epoch = smp.utils.train.TrainEpoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=device,
            verbose=True
        )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True
    )

    max_score = 0
    for i in range(0, epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_dl)
        valid_logs = valid_epoch.run(valid_dl)

        if max_score < valid_logs["fscore"]:
            max_score = valid_logs["fscore"]
            torch.save(model, config.get("model_out_pth").format(date=str(date.today())))
            print("Model saved!")
                       
        train_loss.append(train_logs["dice_loss"])
        valid_loss.append(valid_logs["dice_loss"])
        train_fscore.append(train_logs["fscore"])
        valid_fscore.append(valid_logs["fscore"])
        train_iou.append(train_logs["iou_score"])
        valid_iou.append(valid_logs["iou_score"])  


    return train_loss, valid_loss, train_fscore, valid_fscore, train_iou, valid_iou                                             

    
                                                                                