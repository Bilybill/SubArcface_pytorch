import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter


class SubArcFace(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        scale,
        margin,
        subcenters,
        with_theta,
        clip_thresh,
        clip_value,
        with_weight,
        fc_std,
        if_clip,
    ):
        super(SubArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.K = subcenters
        self.with_theta = with_theta
        self.clip_thresh = clip_thresh
        self.clip_value = clip_value
        self.with_weight = with_weight
        self.if_clip = if_clip
        self.pool = nn.MaxPool1d(self.K)
        self.weight = Parameter(torch.Tensor(out_features,self.K, in_features))
        # self.reset_parameters()
        self.weight.data.normal_(std=fc_std) 

        self.thresh = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin 
        self.cnt = 0

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        # self.logits.weight.data.normal_(std=self.config["fc_std"])
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, label):
        ex = input / torch.norm(input, 2, 1, keepdim=True)
        ew = self.weight / torch.norm(self.weight, 2, 2, keepdim=True)
        cos = torch.matmul(ew,ex.t())
        cos = cos.permute(2,0,1)
        if self.with_theta:
            non_pool_cos = torch.clone(cos)
        cos = self.pool(cos).squeeze()

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
            non_pool_thetas = torch.masked_select(non_pool_cos.acos_() * 180 / math.pi,(a>0)[:,:,None].expand_as(non_pool_cos)).view(input.size(0),self.K)
            if self.with_weight:
                return {
                    "logits": logits,
                    "thetas": thetas,
                    "weight": self.weight,
                    "non_pool_theta": non_pool_thetas,
                }
            else:
                return {
                    "logits": logits,
                    "thetas": thetas,
                    "non_pool_theta": non_pool_thetas,
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
