import os
import logging
import yaml
import numpy as np
from easydict import EasyDict as edict

import torch
from torch.utils.checkpoint import checkpoint


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length):
        self.length = length
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0

    def empty(self):
        return len(self.history) == 0

    def update(self, val):
        self.history.append(val)
        if self.length > 0 and len(self.history) > self.length:
            del self.history[0]

        self.val = val
        self.avg = np.mean(self.history)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_state(state, path, step):
    path, filename = os.path.split(path)
    assert path != ""
    if not os.path.exists(path):
        os.makedirs(path)
    print("saving to {}/{}_iter_{}.pth.tar".format(path, filename, step))
    torch.save(state, "{}/{}_iter_{}.pth.tar".format(path, filename, step))


def load_last_iter(path):
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location="cpu")
        print("=> loaded last_iter={} from {}".format(checkpoint["step"], path))
        return checkpoint["step"]
    else:
        raise RuntimeError("=> no checkpoint found at {}".format(path))


def load_state(path, model, ignore=[], optimizer=None, cuda=False):
    def map_func_cuda(storage, location):
        return storage.cuda()

    def map_func_cpu(storage, location):
        return storage.cpu()

    if cuda:
        map_func = map_func_cuda
    else:
        map_func = map_func_cpu

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        keys1 = set(checkpoint["state_dict"].keys())
        keys2 = set([k for k, _ in model.named_parameters()])
        not_loaded = keys2 - keys1
        for k in not_loaded:
            print("caution: {} not loaded".format(k))

        if optimizer != None:
            assert len(ignore) == 0
            optimizer.load_state_dict(checkpoint["optimizer"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    try:
                        state[k] = v.cuda()
                    except:
                        print("{} can not be set as cuda", k)

            print(
                "=> loaded checkpoint '{}' (step {})".format(path, checkpoint["step"])
            )
            return checkpoint["step"]
    else:
        assert False, "=> no checkpoint found at '{}'".format(path)


def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)20s][line:%(lineno)4d][%(levelname)8s] %(message)s"
    )
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l


class Config(object):
    def __init__(self, config_file):

        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.config_path = config_file

        train_data = config["train_data"]

        self.config = config
        self.config_file = config_file
        self.train_data = train_data
