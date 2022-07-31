import os
import yaml
import torch
import logging
from logging import getLogger

def setup_logger(cfg, name="log"):
    logger = getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))

    ch2 = logging.FileHandler(filename=os.path.join(cfg.save_path, "log.txt"))
    ch2.setLevel(logging.DEBUG)
    ch2.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(ch)
    logger.addHandler(ch2)
    logger.info(f'The CKPT saved here: {cfg.save_path}')

    for i, val in cfg.__dict__.items():
        logger.info(f"{i} : {val}")
    
    return logger


def save_config(cfg, path):
    with open(path, "w") as f:
        yaml.dump(cfg.__dict__, f)


def save_checkpoint(epoch, model, optimizer, path, beta, logger=None):
    save_path = os.path.join(path, "model_final_{}.pth".format(beta))
    if logger:
        logger.info("Saving checkpoint to {}".format(save_path))
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict,
    }, save_path)