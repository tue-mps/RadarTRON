import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling.backbone.resnet import CNNBlockBase
from detectron2.layers import Conv2d, get_norm
import fvcore.nn.weight_init as weight_init

class MIMOStem(CNNBlockBase):
    """
    A custom stem that replicates the logic of MIMO_PreEncoder.
    This includes:
      - Wrap-around cat in the width dimension,
      - Convolution with specific kernel/dilation,
      - Center-cropping to restore original width,
      - Optional batch norm (depends on config),
      - Possibly no downsampling (stride=1), or your own stride choice.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 12),
        dilation=(1, 16),
        use_bn=False,
        padding_ants=96,   # your custom wrap-around size
        stride=1,
        norm=""
    ):
        """
        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: (H, W) size of the convolution kernel
            dilation: (H, W) dilation
            use_bn: whether to apply batch norm after the conv
            padding_ants: how many columns to wrap around on each side
            stride: overall stride you'd like to report to Detectron2
            norm: name of the normalization layer (if using detectron2's get_norm)
        """

        # We call CNNBlockBase.__init__ to tell detectron2 the "logical" stride & channels of this block.
        super().__init__(in_channels, out_channels, stride)
        
        self.use_bn = use_bn
        self.padding_ants = padding_ants  # how many columns to wrap
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),    # Usually MIMO_PreEncoder doesn't do stride>1
            dilation=dilation,
            padding=(1,0),        # no standard conv padding, since we do wrap-around ourselves
            bias=not use_bn,
            norm=get_norm(norm, out_channels) if use_bn else None,
        )
        weight_init.c2_msra_fill(self.conv)

        if use_bn:
            # If you need a separate BN as in MIMO_PreEncoder, you could do so
            # But get_norm(norm, out_channels) might already handle it. Adjust as needed.
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        1) Wrap-around cat on width dimension,
        2) Convolution,
        3) Center-crop,
        4) Optional BN.
        """
        B, C, H, W = x.shape

        # 1) Wrap-around along width dimension
        #    x[..., -padding:] is last columns,
        #    x[..., :padding] is first columns
        x = torch.cat([x[..., -self.padding_ants:], x, x[..., :self.padding_ants]], dim=3)

        # 2) Convolution
        x = self.conv(x)

        # 3) Center-crop to restore original width
        # after wrapping, new width = W + 2*padding_ants - (kernel_width - 1)*(dilation - 1) etc.
        # but let's do a simpler approach like MIMO_PreEncoder does: 
        # x[..., int(x.shape[-1]/2 - W/2) : int(x.shape[-1]/2 + W/2)]
        # Make sure it’s integer-safe for odd/even widths.
        outW = x.shape[-1]
        left = int(outW//2 - W//2)
        right = left + W
        x = x[..., left:right]

        # 4) Optional BN
        if self.use_bn:
            x = self.bn(x)
        
        # 5) Optionally a ReLU here or let upstream do it
        x = F.relu_(x)
        return x