import torch
import math
import numpy as np
import os
import matplotlib.pyplot as plt


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)


def plot_grad_flow(model, epoch, dir):
    named_parameters = model.named_parameters()
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
    plt.figure(figsize=(8, 6))
    plt.plot(ave_grads, color="b", marker="o")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads)), layers, rotation="vertical", fontsize=8)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.tight_layout()
    file_name = f"grad_flow_{epoch}.png"
    plt.savefig(dir + file_name, dpi=100)
    plt.close()


def get_inner_model(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def load_map_from_file(filename):
    with open(filename, 'r') as file:
        map_2d = [line.strip() for line in file.readlines()[4:]]
    map_1d = []
    for row in map_2d:
        for cell in row:
            if cell == '@':
                map_1d.append(1)
            else:
                map_1d.append(0)
    return np.array(map_1d)
