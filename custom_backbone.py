import torch.nn as nn
import torch
from detectron2.modeling.backbone import build_resnet_backbone, Backbone, build_backbone
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from detectron2.modeling.backbone.fpn import FPN
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone

class CustomBackboneWrapper(Backbone):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def output_shape(self):
        shapes = self.backbone.output_shape()
        # Create new shapes dictionary while forcing specific stride values:
        new_shapes = {}
        if "res2" in shapes:
            # Force to original stride 4 (so that FPN computes log2(4)=2 -> "p2")
            new_shapes["res2"] = ShapeSpec(
                channels=shapes["res2"].channels,
                stride=1,  # force to 4
                height=shapes["res2"].height,
                width=shapes["res2"].width,
            )
        if "res3" in shapes:
            new_shapes["res3"] = ShapeSpec(
                channels=shapes["res3"].channels,
                stride=2,  # force to 8
                height=shapes["res3"].height,
                width=shapes["res3"].width,
            )
        if "res4" in shapes:
            new_shapes["res4"] = ShapeSpec(
                channels=shapes["res4"].channels,
                stride=4,  # force to 16
                height=shapes["res4"].height,
                width=shapes["res4"].width,
            )

        if "res5" in shapes:
            new_shapes["res5"] = ShapeSpec(
                channels=shapes["res5"].channels,
                stride=8,  # force to 32
                height=shapes["res5"].height,
                width=shapes["res5"].width,
            )
        return new_shapes


from MIMOStem import MIMOStem
@BACKBONE_REGISTRY.register()
class CustomResNetBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()  # Properly initialize the parent class
        # Build a standard ResNet backbone from the config with the given input_shape.
        bottom_up = build_resnet_backbone(cfg, input_shape)

        bottom_up.stem = MIMOStem(
            #in_channels=192,
            in_channels=input_shape.channels,
            out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
            kernel_size=(3,12),
            dilation=(1,16),
            use_bn=True,  # or not, as you wish
            padding_ants=96,  # or something suiting your data
            stride=1,         # or 2, if you want downsampling
            norm=cfg.MODEL.RESNETS.NORM
        )
        bottom_up = CustomBackboneWrapper(bottom_up) 

        # Now, use the same configuration parameters:
        in_features = cfg.MODEL.FPN.IN_FEATURES
        out_channels = cfg.MODEL.FPN.OUT_CHANNELS

        # Manually build the FPN with top_block set to None to disable extra levels (p6)
        backbone = FPN(
            bottom_up=bottom_up,
            in_features=in_features,
            out_channels=out_channels,
            norm=cfg.MODEL.FPN.NORM,
            top_block=None,  # Disable the top block that would normally generate p6
            fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        )
        
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def output_shape(self):
        return self.backbone.output_shape()

