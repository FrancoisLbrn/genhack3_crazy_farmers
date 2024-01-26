import os, json, shutil, types
import numpy as np
from os.path import join, exists
import pandas as pd



class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def prepare_dirs(trial_dir, flush):
    if not exists(trial_dir):
        os.makedirs(trial_dir)
    elif flush:
        shutil.rmtree(trial_dir)
        if not exists(trial_dir):
            os.makedirs(trial_dir)


def save_config(trial_dir, config):
    config_path = join(trial_dir, f'config_r{config.repeat_num}s{config.split_num}.json')

    all_params = config.__dict__
    with open(config_path, 'w') as fp:
        json.dump(all_params, fp, indent=4, sort_keys=False)


import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import PIL

def apply_colormap(image, colormap=plt.cm.jet):
    # Normalize the image to be in the range [0, 1]
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Apply colormap
    colored_image = colormap(normalized_image)

    # Convert to RGB and remove the alpha channel
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)

    # Convert to PIL Image
    pil_image = PIL.Image.fromarray(colored_image)

    # Convert to PyTorch tensor
    tensor_image = transforms.ToTensor()(pil_image)

    return tensor_image  