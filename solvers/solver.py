import os
import time
import random

from easydict import EasyDict as edict
from tensorboardX import SummaryWriter
from thop import profile

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from core.data import FaceDataset
import core.models as models
import core.losses as losses
from core.samplers import GivenIterationSampler
from core.lr_schedulers import lr_scheduler_entry
from core.utils import (
    AverageMeter,
    accuracy,
    load_state,
    load_last_iter,
    save_state,
    create_logger,
)


class Solver:
    def __init__(self, C):
        self.C = C
        config = edict(C.config["common"])
        self.config = config

        save_path = config.get("save_path", os.path.dirname(C.config_file))
        print("--save--path is {}".format(save_path))

        if not os.path.exists("{}/events".format(save_path)):
            os.makedirs("{}/events".format(save_path))
        if not os.path.exists("{}/logs".format(save_path)):
            os.makedirs("{}/logs".format(save_path))
        if not os.path.exists("{}/checkpoints".format(save_path)):
            os.makedirs("{}/checkpoints".format(save_path))
        self.tb_logger = SummaryWriter("{}/events".format(save_path))
        self.logger = create_logger(
            "global_logger", "{}/logs/log.txt".format(save_path)
        )

        if self.config.get("deterministic", "True"):
            cudnn.deterministic = True
            cudnn.benchmark = False
            random.seed(210)
            np.random.seed(210)
            torch.manual_seed(210)
            torch.cuda.manual_seed(210)

        self.feature_dim = self.config.model["kwargs"]["feature_dim"]
        self.save_path = save_path
        self.last_iter = -1

        self.last_state_dict = {}
        self.last_optim_state_dict = {}
        self.last_save_iter = -1

        tmp = edict()
        self.tmp = tmp

    def initialize(self, args):

        self.create_dataset()
        self.create_model()
        self.create_optimizer()

        if args.recover:
            self.last_iter = load_last_iter(args.load_path)
            self.last_iter -= 1
        self.load_args = args
        self.create_dataloader()
        self.create_lr_scheduler()

    def create_dataset(self):
        self.dataset = FaceDataset(self.C.config)

    def create_dataloader(self):
        config = self.config
        self.sampler = GivenIterationSampler(
            self.dataset, config.max_iter, config.batch_size, last_iter=self.last_iter
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
            pin_memory=False,
            sampler=self.sampler,
        )

    def create_model(self):
        config = self.config

        model = models.model_entry(config.model)
        input = torch.randn(1, 3, 224, 224)
        macs, params = profile(model, inputs=(input,))
        print("macs: {}, params: {}".format(macs / (1024 ** 2), params / (1024 ** 2)))
        loss_args = config.loss["kwargs"]
        loss_args["feature_dim"] = self.feature_dim
        loss_args["num_classes"] = self.C.config["train_data"]["num_classes"]
        loss_args["base"] = model
        model = losses.loss_entry(config.loss)
        model.cuda()
        self.model = model
        self.smloss = torch.nn.CrossEntropyLoss()

    def create_optimizer(self):
        config = self.config

        nesterov = config.get("nesterov", False)
        optim_method = config.get("optim", "sgd").lower()

        if optim_method == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                config.lr_scheduler.kwargs.base_lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                nesterov=nesterov,
            )
        elif optim_method == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr = config.lr_scheduler.kwargs.base_lr,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError("do not support {} optimizer".format(optim_method))

    def create_lr_scheduler(self):
        self.config.lr_scheduler.kwargs.optimizer = self.optimizer
        self.config.lr_scheduler.kwargs.last_iter = self.last_iter
        self.config.lr_scheduler.kwargs.max_iter = self.config.max_iter
        self.lr_scheduler = lr_scheduler_entry(self.config.lr_scheduler)

    def tb_logging(self):
        tmp = self.tmp
        self.tb_logger.add_scalar("loss", float(tmp.loss.cpu()), tmp.current_step)
        self.tb_logger.add_scalar("acc", float(tmp.top1.cpu()), tmp.current_step)
        self.tb_logger.add_scalar("lr", tmp.current_lr, tmp.current_step)

    def logging(self):
        tmp = self.tmp
        config = self.config
        self.logger.info(
            "Iter: [{0}/{1}]\t"
            "Time {batch_time.avg:.3f} ({data_time.avg:.3f})\t"
            "Loss {loss:.4f}\t"
            "Prec@1 {top1:.3f}\t"
            "LR {current_lr}".format(
                tmp.current_step,
                config.max_iter,
                batch_time=tmp.vbatch_time,
                data_time=tmp.vdata_time,
                loss=float(tmp.loss.cpu()),
                top1=float(tmp.top1.cpu()),
                current_lr=tmp.current_lr,
            )
        )

    def load(self, args):
        config = self.config
        if args.load_path == "":
            return
        if args.recover:
            self.last_iter = load_state(
                args.load_path, self.model, optimizer=self.optimizer
            )
            self.last_iter -= 1
        else:
            load_state(args.load_path, self.model)

    def pre_run(self):
        tmp = self.tmp
        tmp.vbatch_time = AverageMeter(10)
        tmp.vdata_time = AverageMeter(10)
        self.model.train()

    def prepare_data(self):
        tmp = self.tmp
        tmp.input_var = dict()
        for k, v in tmp.input.items():
            if not isinstance(v, list):
                tmp.input_var[k] = v.cuda()

    def run(self):
        config = self.config
        tmp = self.tmp

        self.pre_run()

        end = time.time()

        load_flag = True

        for i, tmp.input in enumerate(self.loader):
            tmp.vdata_time.update(time.time() - end)

            self.prepare_data()

            if load_flag:
                tmp.current_step = 0
                self.load(self.load_args)
                load_flag = False

            tmp.current_step = self.last_iter + i + 1
            self.lr_scheduler.step(tmp.current_step)
            tmp.current_lr = self.lr_scheduler.get_lr()[0]
            
            output = self.model(tmp.input_var["image"], tmp.input_var["label"])
            tmp.loss = self.smloss(output["logits"], tmp.input_var["label"])
            tmp.top1 = accuracy(
                output["logits"].data, tmp.input["label"].cuda(), topk=(1, 5)
            )[0]

            self.optimizer.zero_grad()
            tmp.loss.backward()
            self.optimizer.step()

            tmp.vbatch_time.update(time.time() - end)
            end = time.time()

            if tmp.current_step % config.print_freq == 0:
                self.tb_logging()
                self.logging()

            if (
                config.save_interval > 0
                and (tmp.current_step + 1) % config.save_interval == 0
            ):
                ckpt_name = save_state(
                    {
                        "step": tmp.current_step + 1,
                        "model_args": config.model,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    "{}/checkpoints/ckpt".format(self.save_path),
                    tmp.current_step + 1,
                )
