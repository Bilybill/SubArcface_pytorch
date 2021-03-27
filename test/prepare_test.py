import os
import sys
import numpy as np
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler

from core.utils import Config
import core.models as models
from core.data import SimpleDataset
from core.utils import AverageMeter, load_state


class SimpleFeature(nn.Module):
    def __init__(self, base, output_name):
        super(SimpleFeature, self).__init__()
        self.base = base
        self.output_name = output_name

    def forward(self, x):
        x = self.base(x)
        return x[self.output_name]


def prepare_test(args):

    config_path = args.config_path
    model_path = args.model_path

    C = Config(config_path)

    aug_config = C.config["augmentation"]
    test_dataset = SimpleDataset(
        args.prefix,
        args.list_file,
        aug_config["crop_size"],
        aug_config["final_size"],
        aug_config["crop_center_y_offset"],
    )
    sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        sampler=sampler,
    )

    m_config = C.config["common"]["model"]
    print("=> creating model '{}'".format(m_config["type"]))
    model = models.model_entry(m_config)
    model = SimpleFeature(model, m_config["test"]["output_name"])
    model.cuda()
    model.eval()
    load_state(model_path, model)
    return model, test_loader


def test_forward(model, input):
    return model(input.cuda())
