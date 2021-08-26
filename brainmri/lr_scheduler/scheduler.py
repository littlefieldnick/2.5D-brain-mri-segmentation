import torch.optim.lr_scheduler as lr_scheduler

def step_lr(opt, last_epoch, step_size, gamma=0.1):
    return lr_scheduler.StepLR(opt, last_epoch=last_epoch, step_size=step_size, gamma=gamma)
   
def multi_step_lr(opt, last_epoch, milestones=[100, 1000], gamme=0.1):
    return lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma, last_epoch=last_epoch)
    
def exponential(opt, last_epoch, gamma=0.995):
    return lr_scheduler.ExponentialLR(opt, gamma=gamma, last_epoch=last_epoch)
    
def reduce_lr_on_plateau(optimizer, last_epoch, mode='min', factor=0.1, patience=10,
                         threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0):
    return lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                                        threshold=threshold, threshold_mode=threshold_mode,
                                        cooldown=cooldown, min_lr=min_lr)

def cosine(optimizer, last_epoch, T_max=50, eta_min=0.00001):
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min,
                                        last_epoch=last_epoch)