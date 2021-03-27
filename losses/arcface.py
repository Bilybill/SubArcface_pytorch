import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter


class ArcFace(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        scale,
        margin,
        with_theta,
        clip_thresh,
        clip_value,
        with_weight,
    ):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.with_theta = with_theta
        self.clip_thresh = clip_thresh
        self.clip_value = clip_value
        self.with_weight = with_weight
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

        self.thresh = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin# ???
        self.cnt = 0

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, label):
        ex = input / torch.norm(input, 2, 1, keepdim=True)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        cos = torch.mm(ex, ew.t())
        a = torch.zeros_like(cos)
        thetas = []

        # # is it necessary to do this loop?
        # for i in range(a.size(0)):
        #     lb = int(label[i])
        #     if cos[i, lb].item() > self.clip_thresh: #1.5
        #         cos[i, lb] = self.clip_value

        b = torch.zeros_like(cos)

        # I can not stand this stupid code any more 
        for i in range(a.size(0)):
            lb = int(label[i])
            theta = math.acos(cos[i, lb].item()) / math.pi * 180
            thetas.append(theta)
            if cos[i, lb].item() > self.thresh:
                a[i, lb] = a[i, lb] + self.margin
            else:
                b[i, lb] = b[i, lb] - self.mm

        if self.with_theta:
            if self.with_weight:
                return {
                    "logits": self.scale * (torch.cos(torch.acos(cos) + a) + b),
                    "thetas": thetas,
                    "weight": self.weight,
                }
            else:
                return {
                    "logits": self.scale * (torch.cos(torch.acos(cos) + a) + b),
                    "thetas": thetas,
                }
        else:
            if self.with_weight:
                return {
                    "logits": self.scale * (torch.cos(torch.acos(cos) + a) + b),
                    "weight": self.weight,
                }
            else:
                return {"logits": self.scale * (torch.cos(torch.acos(cos) + a) + b)}

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features="
            + str(self.in_features)
            + ", out_features="
            + str(self.out_features)
            + ", scale="
            + str(self.scale)
            + ", margin="
            + str(self.margin)
            + ")"
        )
