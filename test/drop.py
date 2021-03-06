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

parser = argparse.ArgumentParser(description="Drop non-dominant data")
parser.add_argument("--prefix", default="", type=str)
parser.add_argument("--list-file", default="", type=str)
parser.add_argument("--meta-file",default="",type=str)
parser.add_argument("--config-path", default="", type=str)
parser.add_argument("--model-path", default="", type=str)
parser.add_argument("--save-path", default="", type=str)
args = parser.parse_args()


def prepare_drop(args):
    config_path = args.config_path
    model_path = args.model_path

    C = Config(config_path)
    if C.config["train_data"]["drop_mode"] == False:
        print("Change mode into drop mode")
        C.config["train_data"]["drop_mode"] = True
    train_data_info = C.config["train_data"]
    train_data_info["prefix"][0] = args.prefix
    train_data_info["meta"][0] = args.meta_file
    train_data_info["list"][0] = args.list_file

    drop_dataset = FaceDataset(C.config)
    drop_dataloader = DataLoader(
        drop_dataset,
        batch_size = C.config["common"]["batch_size"],
        shuffle = False,
        num_workers = 3,
        pin_memory = False,
        sampler = SequentialSampler(drop_dataset)
    )

    m_config = C.config["common"]["model"]
    print("=> creating model '{}'".format(m_config["type"]))
    model = models.model_entry(m_config)
    config = edict(C.config["common"])
    loss_args = config.loss["kwargs"]
    loss_args["feature_dim"] = m_config["kwargs"]["feature_dim"]
    loss_args["num_classes"] = C.config["train_data"]["num_classes"]
    loss_args["base"] = model
    loss_args["weight"] = True
    loss_args["with_theta"] = True
    loss_args["type"] = "SubArcFace"
    model = losses.loss_entry(config.loss)
    # for k, v in model.named_parameters():
    #     print(v.size(),k)
    # print(set([v for k, v in model.named_parameters()]))
    model.cuda()
    model.eval()
    load_state(model_path,model)
    
    return model,drop_dataloader

def drop_forward(model,input_data):
    # print(input_data)
    # raise ValueError("stop here")
    input_var = dict()
    for k, v in input_data.items():
        if not isinstance(v, list):
            input_var[k] = v.cuda()
    # print(input_var["image"].size())
    return model(input_var["image"],input_var["label"])

def drop():

    torch.cuda.set_device(3)

    model, drop_loader = prepare_drop(args)

    # input_test = torch.randn(1, 3, 224, 224).cuda()
    # macs, params = profile(model, inputs=(input_test,))

    batch_time = AverageMeter(10)
    data_time = AverageMeter(10)

    thetas = []
    # weight = []
    non_pooltheta = []
    with torch.no_grad():
        end = time.time()
        for i, input_data in enumerate(drop_loader):
            data_time.update(time.time() - end)
            output = drop_forward(model, input_data)
            thetas.append(output["thetas"].cpu().numpy())
            weight=output["weight"].cpu().numpy()
            non_pooltheta.append(output["non_pool_theta"].cpu().numpy())
            # features.append(output.data.cpu().numpy())
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 10 == 0:
                print(
                    "Extract: [{0}/{1}]\t"
                    "Time {batch_time.avg:.3f} ({data_time.avg:.3f})".format(
                        i, len(drop_loader), batch_time=batch_time, data_time=data_time
                    )
                )

    thetas = np.concatenate(thetas, axis=0)
    # weight = np.concatenate(weight, axis=0)
    non_pooltheta = np.concatenate(non_pooltheta, axis=0)
    return thetas, weight, non_pooltheta


if __name__ == "__main__":
    prepare_drop(args)
    '''
    thetas, weight,non_pooltheta = drop()
    # print(macs / 1024 / 1024)
    dirname = "./theta_weight_v2/"
    dump_path = os.path.join(dirname, args.save_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    np.save("./theta_weight_v2/thetas",thetas)
    np.save("./theta_weight_v2/weight",weight)
    np.save("./theta_weight_v2/non_pooltheta",non_pooltheta)
    '''
    # features.tofile(dump_path)
    # print("finish feature dump!")

