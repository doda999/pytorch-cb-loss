import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxCrossEntorpyLoss(nn.Module):
    def __init__(self, weight=None):
        super(SoftmaxCrossEntorpyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction="sum")
    
    def forward(self, out, label):
        loss_sum = self.criterion(out, label)
        return loss_sum/out.shape[0]


class SigmoidCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(SigmoidCrossEntropyLoss, self).__init__()
        self.weight = weight
    
    def forward(self, out, label):
        weight = None
        if self.weight is not None:
            weight = self.weight[label].view(-1,1).repeat(1, out.shape[1]).cuda()
        one_hot = F.one_hot(label, num_classes=out.shape[1]).to(torch.float)
        loss = F.binary_cross_entropy_with_logits(out, one_hot, weight=weight, reduction='none')
        return loss.sum()/out.shape[0]


class FocalLoss(nn.Module):
    def __init__(self, gamma, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, out, label):
        weight = None
        if self.weight is not None:
            weight = self.weight[label].view(-1,1).repeat(1, out.shape[1]).cuda()
        one_hot = F.one_hot(label, num_classes=out.shape[1]).to(torch.float)
        logpt = F.binary_cross_entropy_with_logits(out, one_hot, weight=weight, reduction='none')
        max_val = torch.clamp(-out, min=0)  # prevent the output of torch.exp from being inf
        focal_weight = torch.exp(-self.gamma * one_hot * out - self.gamma * (torch.log(torch.exp(1. - max_val) + torch.exp(-out - max_val)) + max_val))
        loss = focal_weight * logpt
        return loss.sum()/out.shape[0]


class CombinedLoss(nn.Module):
    def __init__(self, cfg, weight=None):
        super(CombinedLoss, self).__init__()
        self.weight_decay = cfg.weight_decay
        self.loss_type = cfg.loss_type
        if cfg.loss_type == "softmax":
            self.cls_loss = SoftmaxCrossEntorpyLoss(weight)
        elif cfg.loss_type == "sigmoid":
            self.cls_loss = SigmoidCrossEntropyLoss(weight)
        else:
            self.cls_loss = FocalLoss(cfg.gamma, weight)
        self.cls_loss.cuda()
    
    def forward(self, out, label, named_parameters):
        loss_items = {}
        loss_items["classification"] = self.cls_loss(out, label)
        loss_items["regularization"] = self.weight_decay * sum([torch.norm(param) for (name, param) in named_parameters if self.loss_type == "softmax" or "linear.bias" not in name])
        return loss_items


def get_loss_func(cfg, num_per_cls):
    weight = None
    if cfg.beta:
        weight = (1. - cfg.beta) / (1. - torch.pow(cfg.beta, torch.tensor(num_per_cls)))
        weight = weight / torch.sum(weight) * len(num_per_cls)
    return CombinedLoss(cfg, weight)

