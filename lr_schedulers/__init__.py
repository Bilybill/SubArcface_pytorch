from .steplr_scheduler import StepLRScheduler


def lr_scheduler_entry(config):
    return globals()[config["type"] + "LRScheduler"](**config["kwargs"])
