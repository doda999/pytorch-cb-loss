import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from src.config import Config
from src.utils import setup_logger, save_config, save_checkpoint
from src.resnet import build_model
from src.data import CifarDataset
from src.loss import get_loss_func
from src.scheduler import WarmupStepScheduler
from test import evaluate

def train(cfg, logger):
    # load data
    dataset = CifarDataset(cfg, mode="train")
    # data loader
    data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    num_per_cls = dataset.get_num_per_cls()
    logger.info("Data Distribution Detail (class : number)")
    data_info = ""
    for cls, num in enumerate(num_per_cls):
        data_info += "{} : {}".format(cls, num)
        if cls != dataset.num_labels - 1:
            data_info += "\n" 
    logger.info(data_info)

    # build model
    model = build_model(cfg.model, dataset.num_labels, cfg.loss_type)
    if cfg.checkpoint:
        model.load_state_dict(torch.load(cfg.checkpoint, map_location=torch.device("cpu"))['model'])
    
    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    if cfg.lr_warmup:
        scheduler = WarmupStepScheduler(optimizer, milestones=cfg.lr_decay_step, gamma=cfg.lr_decay, warmup_epoch=cfg.lr_warmup_epoch)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.lr_decay_step, gamma=cfg.lr_decay)

    # loss func
    loss_func = get_loss_func(cfg, num_per_cls)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    model.train()
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(cfg.num_epoch):
        epoch_cls_loss = 0
        epoch_reg_loss = 0
        cnt = 0
        for (image, label) in data_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            out = model(image)
            loss_items = loss_func(out, label, model.named_parameters())
            losses = sum(loss for loss in loss_items.values())
            losses.backward()
            optimizer.step()
            epoch_cls_loss += loss_items["classification"]
            epoch_reg_loss += loss_items["regularization"]
            cnt += 1
        scheduler.step()
        logger.info("Epoch {}: classification loss - {:.6f} regularization loss - {:.6f} lr - {:.6f}".format(epoch+1, epoch_cls_loss/cnt, epoch_reg_loss/cnt, optimizer.param_groups[-1]["lr"]))
        if epoch == cfg.num_epoch - 1:
            save_checkpoint(epoch, model, optimizer, cfg.save_path, cfg.beta, logger=logger)

def main():
    cfg = Config()
    if not os.path.exists(cfg.save_path):
        os.mkdir(cfg.save_path)
    save_config(cfg, os.path.join(cfg.save_path, "config.yaml"))
    logger = setup_logger(cfg, name="train")
    # train
    train(cfg, logger=logger)
    if not cfg.no_eval:
        evaluate(cfg, logger)


if __name__ == "__main__":
   main()