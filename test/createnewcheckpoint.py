import os
import sys
import argparse
import numpy as np
import time
from thop import profile
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from easydict import EasyDict as edict

import torch
import torch.nn as nn

sys.path.append("../")
from utils import AverageMeter
from data import FaceDataset
import models as models
import losses as losses
from utils import (
    AverageMeter,
    accuracy,
    load_state,
    load_last_iter,
    save_state,
    create_logger,
    Config,
)

parser = argparse.ArgumentParser(description="create new checkpoint with drop weights")
parser.add_argument("--config-path", default="", type=str)
parser.add_argument("--model-path", default="", type=str)
parser.add_argument("--save-path", default="", type=str)
args = parser.parse_args()


def prepare_model(args):
    config_path = args.config_path
    model_path = args.model_path

    C = Config(config_path)

    m_config = C.config["common"]["model"]
    print("=> creating model '{}'".format(m_config["type"]))
    model = models.model_entry(m_config)
    config = edict(C.config["common"])
    loss_args = config.loss["kwargs"]
    loss_args["feature_dim"] = 512
    loss_args["num_classes"] = 1000
    loss_args["base"] = model
    loss_args["weight"] = False
    loss_args["with_theta"] = False
    loss_args["type"] = "SubArcFace"
    model = losses.loss_entry(config.loss)
    return model

def createnewcheckpoint():
    def map_func_cpu(storage, location):
        return storage.cpu()
    mapfunc = map_func_cpu
    # torch.cuda.set_device(3)

    model = prepare_model(args)
    path = args.model_path

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func_cpu)
        model_dict = model.state_dict()
        pre_traindict = {k:v for k,v in checkpoint['state_dict'].items() if k in model_dict.keys() and v.size()==model_dict[k].size()}
        keys1 = set(pre_traindict.keys())
        keys2 = set(model_dict.keys())
        keys = keys2 - keys1
        reweight = np.load("../losses/subarc_weight/weight_drop.npy")
        reweight = torch.from_numpy(reweight)
        we = {list(keys)[0]:reweight}
        # print(model_dict[list(keys)[0]])
        model_dict.update(pre_traindict)
        model_dict.update(we)
        model.load_state_dict(model_dict)
        # print(model.state_dict()[list(keys)[0]])


        # model.load_state_dict(checkpoint["state_dict"], strict=False)
        # keys1 = set(checkpoint["state_dict"].keys())
        # keys2 = set([k for k, _ in model.named_parameters()])
        # not_loaded = keys2 - keys1
        # for k in not_loaded:
        #     print("caution: {} not loaded".format(k))
        ckpt_name = save_state(
            {
                "step":-1,
                "state_dict":model.state_dict(),
            },
            "{}/checkpoints/ckpt".format(args.save_path),
            0,
        )
    else:
        assert False, "=> no checkpoint found at '{}'".format(path)



if __name__ == "__main__":
    createnewcheckpoint()
