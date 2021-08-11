from data_gen.slices import MriStacker
from data_gen.dataset import BrainMriSegmentation, get_dataloader, get_augmentations
from utils.config import *
import argparse

import torch
import torch.optim as optim

import segmentation_models_pytorch as smp

def get_parser():
    parser = argparse.ArgumentParser(description="Training configuration file")
    parser.add_argument("--config", dest="config_file",
                        help="Location of the configuration file for training.")
    return parser

def train_model(config, train_dl, valid_dl):
    model_type = config.get_config_setting("model")
    encoder = config.get_config_setting("model_encoder")
    encoder_weights = config.get_config_setting("encoder_weights")
    activation = config.get_config_setting("model_act")
    classes = config.get_config_setting("num_classes")
    lr = config.get_config_setting("lr")
    epochs = config.get_config_setting("epochs")

    model = None
    if model_type == "unet":
        model = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                         classes=classes, activation=activation)
    elif model_type == "fpn":
        model = smp.FPN(encoder_name=encoder, encoder_weights=encoder_weights,
                         classes=classes, activation=activation)
    elif model_type == "unet++":
        model = smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=encoder_weights,
                         classes=classes, activation=activation)
    else:
        print("Model type is not valid. Training cannot be done...")
        return

    loss = smp.utils.losses.DiceLoss()
    metrics = [smp.utils.metrics.IoU(threshold=0.5)]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if torch.cuda.device_count() > 1 and config.get_config_setting("device") == "cuda":
        print("Training using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).cuda()

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=config.get_config_setting("device"),
        verbose=True
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=config.get_config_setting("device"),
        verbose=True
    )

    lr_scheduler = None
    if config.get_config_setting("use_lr_scheduler"):
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                           max_lr=lr/10,
                                                           steps_per_epoch=len(train_dl),
                                                           epochs=epochs)

    max_score = 0
    for i in range(0, epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_dl)
        valid_logs = valid_epoch.run(valid_dl)

        if max_score < valid_logs["iou_score"]:
            max_score = valid_logs["iou_score"]
            torch.save(model, config.get_config_setting("model_out_pth"))
            print("Model saved!")

        if lr_scheduler is not None:
            lr_scheduler.step()

def train_runner(config):

    stacker = MriStacker(root_dir=config.get_config_setting("data_dir"),
                         out_dir=config.get_config_setting("out_dir"),
                         stack_size=config.get_config_setting("stack_size"))

    if config.get_config_setting("make_stacks"):
        stacker.process_patients()

    stacker.gen_train_val_test_split()

    train_ds = BrainMriSegmentation(stacker.train_df, config.get_config_setting("stack_size"),
                                    transforms=get_augmentations(is_train=True, apply_transforms=config.get_config_setting("augmentations")))
    valid_ds = BrainMriSegmentation(stacker.valid_df, config.get_config_setting("stack_size"),
                                    transforms=get_augmentations(is_train=False))

    train_dl = get_dataloader(train_ds, bs=config.get_config_setting("batch_size"))
    valid_dl = get_dataloader(valid_ds, bs=config.get_config_setting("batch_size"))

    train_model(config, train_dl, valid_dl)

def main():
    parser = get_parser()
    args = parser.parse_args()

    config = None
    if args.config_file is None:
        print("No configuration file was provided. Exiting...")
        exit(0)
    else:
        config = Config(config_file_pth=args.config_file)

    train_runner(config)

if __name__ == "__main__":
    main()