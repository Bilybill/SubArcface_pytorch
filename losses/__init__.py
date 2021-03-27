from .loss_wrapper import Wrapper


def loss_entry(config):
    return globals()[config["type"]](config["kwargs"])
