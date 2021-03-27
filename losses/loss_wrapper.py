import torch.nn as nn
from .arcface import ArcFace
from .arcfacenew import ArcFaceNew
from .subarcface import SubArcFace

class Wrapper(nn.Module):
    def __init__(self, config):
        super(Wrapper, self).__init__()

        self.config = config

        self.base = config["base"]
        feature_dim = config["feature_dim"]
        num_classes = config["num_classes"]

        if self.config["type"] == "ArcFace":
            alpha = self.config["alpha"]
            margin = self.config["margin"]
            with_theta = self.config.get("with_theta", False)
            clip_thresh = self.config.get("clip_thresh", 1.5)
            clip_value = self.config.get("clip_value", 0.0)
            with_weight = self.config.get("weight", False)
            fc_std = self.config["fc_std"]
            self.logits = ArcFace(
                feature_dim,
                num_classes,
                alpha,
                margin,
                with_theta,
                clip_thresh,
                clip_value,
                with_weight,
                fc_std,
            )
        elif self.config["type"] == "ArcFaceNew":
            alpha = self.config["alpha"]
            margin = self.config["margin"]
            with_theta = self.config.get("with_theta", False)
            clip_thresh = self.config.get("clip_thresh", 1.5)
            clip_value = self.config.get("clip_value", 0.0)
            with_weight = self.config.get("weight", False)
            fc_std = self.config["fc_std"]
            if_clip = self.config["if_clip"]
            self.logits = ArcFaceNew(
                feature_dim,
                num_classes,
                alpha,
                margin,
                with_theta,
                clip_thresh,
                clip_value,
                with_weight,
                fc_std,
                if_clip,
            )
        elif self.config["type"] == "SubArcFace":
            alpha = self.config["alpha"]
            margin = self.config["margin"]
            with_theta = self.config.get("with_theta", False)
            clip_thresh = self.config.get("clip_thresh", 1.5)
            clip_value = self.config.get("clip_value", 0.0)
            with_weight = self.config.get("weight", False)
            fc_std = self.config["fc_std"]
            if_clip = self.config["if_clip"]
            subcenters = self.config["K"]
            self.logits = SubArcFace(
                feature_dim,
                num_classes,
                alpha,
                margin,
                subcenters,
                with_theta,
                clip_thresh,
                clip_value,
                with_weight,
                fc_std,
                if_clip,
            )
        else:
            raise RuntimeError("unknown loss type {}".format(self.config["type"]))

        # self.logits.weight.data.normal_(std=self.config["fc_std"])#??? why init here??it should be in the init function of Arcface

    def forward(self, x, label=None, feat_only=False):
        x = self.base(x)
        feature = x["feature"]
        feature_nobn = x["feature_nobn"]

        output = {}
        output["feature"] = feature
        output["feature_nobn"] = feature_nobn

        if feat_only:
            return output

        x = self.logits(feature, label)
        output.update(x)

        return output
