from .regnet import *


def model_entry(config):
    return globals()[config["type"]](**config["kwargs"])
