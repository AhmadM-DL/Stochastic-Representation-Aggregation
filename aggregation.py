import torch

def stochastic_aggregation(features):
    mean = features.mean(dim=1)
    variance = features.var(dim=1)
    sampled_representation = mean + torch.randn_like(mean) * torch.sqrt(variance)
    return sampled_representation

def cls_aggregation(features):
    return features[:, 0]

def avg_aggregation(features):
    return features.mean(dim=1)

def get_aggregation_stratey(strategy: str):
    if strategy == "cls":
        return cls_aggregation
    if strategy == "avg":
        return avg_aggregation
    if strategy == "sra":
        return stochastic_aggregation