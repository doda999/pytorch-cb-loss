import os
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

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

def evaluate(cfg, logger):
    # load data
    dataset = CifarDataset(cfg, mode="eval")
    # data loader
    data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
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
    if cfg.checkpoint or os.path.exists(os.path.join(cfg.save_path, "model_final_{}.pth".format(cfg.beta))):
        path = cfg.checkpoint if cfg.checkpoint else os.path.join(cfg.save_path, "model_final_{}.pth".format(cfg.beta))
        model.load_state_dict(torch.load(path, map_location=torch.device("cpu"))['model'])
        logger.info("Model is loaded from {}".format(path))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    correct_labels = [label.item() for label in dataset.labels]
    pred_labels = []
    model.eval()
    with torch.no_grad():
        for (image, label) in data_loader:
            image = image.to(device)
            label = label.to(device)
            out = model(image)
            pred_labels = pred_labels + [pred.item() for pred in torch.argmax(out, dim=1).cpu()]
    
    logger.info("****Evaluation Finished*****")
    acc = accuracy_score(correct_labels, pred_labels)
    logger.info("Accuracy: {:.2f}".format(acc * 100.))
    logger.info("Error Rate: {:.2f}".format((1-acc) * 100.))
    conf_mat = confusion_matrix(correct_labels, pred_labels)
    sns.heatmap(conf_mat, square=True, cbar=True, annot=True, fmt='d', cmap='Blues')
    plt.savefig(os.path.join(cfg.save_path, 'confusion_matrix_{}.png'.format(cfg.beta))) 


def main():
    cfg = Config()
    if not os.path.exists(cfg.save_path):
        os.mkdir(cfg.save_path)
    logger = setup_logger(cfg, name="eval")
    # evaluate
    evaluate(cfg, logger=logger)


if __name__ == "__main__":
   main()