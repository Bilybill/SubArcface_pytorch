import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter


class ArcFaceNew(nn.Module):
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
        fc_std,
        if_clip,
    ):
        super(ArcFaceNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.with_theta = with_theta
        self.clip_thresh = clip_thresh
        self.clip_value = clip_value
        self.with_weight = with_weight
        self.if_clip = if_clip
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        # self.reset_parameters()
        self.weight.data.normal_(std=fc_std)

        self.thresh = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin# ???
        self.cnt = 0

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        # self.logits.weight.data.normal_(std=self.config["fc_std"])
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, label):
        ex = input / torch.norm(input, 2, 1, keepdim=True)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        cos = torch.mm(ex, ew.t())
        index = torch.where(label!=-1)[0]
        if self.if_clip:
            a = torch.zeros_like(cos)
            b = torch.zeros_like(cos)
            a.scatter_(1,label[index,None],self.margin)
            b.scatter_(1,label[index,None],-self.mm)
            mask = (cos>self.thresh)*1
            logits =  torch.cos(cos.acos_() + a * mask) + b * ( 1 - mask )
            logits = self.scale * logits
        else:
            m_hot = torch.zeros_like(cos)
            m_hot.scatter_(1,label[index,None],self.margin)
            cos.acos_()
            cos[index] += m_hot
            cos.cos_().mul_(self.scale)
            logits = cos

        if self.with_theta:
            thetas = torch.masked_select(cos * 180 / math.pi,a>0)
            if self.with_weight:
                return {
                    "logits": logits,
                    "thetas": thetas,
                    "weight": self.weight,
                }
            else:
                return {
                    "logits": logits,
                    "thetas": thetas,
                }
        else:
            if self.with_weight:
                return {
                    "logits": logits,
                    "weight": self.weight,
                }
            else:
                return {"logits": logits}

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
