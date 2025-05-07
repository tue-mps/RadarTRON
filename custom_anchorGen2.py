import torch
from torch import nn
import math
from detectron2.modeling.anchor_generator import ANCHOR_GENERATOR_REGISTRY
from detectron2.config import configurable
from detectron2.structures import Boxes
from detectron2.layers import ShapeSpec
from typing import List
from detectron2.modeling.anchor_generator import BufferList, _broadcast_params
from detectron2.layers import ShapeSpec, move_device_like

@ANCHOR_GENERATOR_REGISTRY.register()
class CustomAnchorGenerator(nn.Module):
    box_dim: torch.jit.Final[int] = 4

    @configurable
    def __init__(self, *, sizes, aspect_ratios, strides, offset=0.5, std_behavior=False):
        super().__init__()
        self.strides = strides
        self.num_features = len(self.strides)
        sizes = _broadcast_params(sizes, self.num_features, "sizes")
        aspect_ratios = _broadcast_params(aspect_ratios, self.num_features, "aspect_ratios")
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios)
        self.offset = offset
        self.std_behavior = std_behavior
        assert 0.0 <= self.offset < 1.0, self.offset

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        return {
            "sizes": cfg.MODEL.ANCHOR_GENERATOR.SIZES,
            "aspect_ratios": cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS,
            "strides": cfg.MODEL.ANCHOR_GENERATOR.STRIDES,
            "offset": cfg.MODEL.ANCHOR_GENERATOR.OFFSET,
            "std_behavior": cfg.MODEL.ANCHOR_GENERATOR.STD_BEHAVIOR,
        }

    def _calculate_anchors(self, sizes, aspect_ratios):
        cell_anchors = [
            self.generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)
        ]
        return BufferList(cell_anchors)

    @property
    def num_cell_anchors(self):
        """
        Alias of `num_anchors`.
        """
        return self.num_anchors

    @property
    def num_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
        anchors = []
        for size in sizes:
            area = size**2.0
            for aspect_ratio in aspect_ratios:
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return torch.tensor(anchors)

    def _create_grid_offsets(self, size: List[int], stride: int, target_device_tensor: torch.Tensor):
        """
        Creates grid offsets based on behavior type
        """
        if not isinstance(size, (list, tuple)):
            raise ValueError(f"Expected size to be a list or tuple, got {type(size)}")
            
        grid_height, grid_width = size[0], size[1]

        if self.std_behavior:
            # Standard behavior: anchors at every position
            shifts_x = move_device_like(
                torch.arange(self.offset * stride, grid_width * stride, step=stride, dtype=torch.float32),
                target_device_tensor,
            )
        else:
            # Center-only behavior: one anchor in the middle
            middle_x = (grid_width * stride) / 2
            shifts_x = move_device_like(
                torch.tensor([middle_x], dtype=torch.float32),
                target_device_tensor,
            )
        
        # Height anchors are the same for both behaviors
        shifts_y = move_device_like(
            torch.arange(self.offset * stride, grid_height * stride, step=stride, dtype=torch.float32),
            target_device_tensor,
        )

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        return shift_x, shift_y

    def _grid_anchors(self, grid_sizes: List[List[int]]):
        """
        Args:
            grid_sizes (List[List[int]]): List of [H, W] for each feature level
        """
        anchors = []
        buffers: List[torch.Tensor] = [x[1] for x in self.cell_anchors.named_buffers()]
        for size, stride, base_anchors in zip(grid_sizes, self.strides, buffers):
            shift_x, shift_y = self._create_grid_offsets(size, stride, base_anchors)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))
        return anchors

    def forward(self, features: List[torch.Tensor]):
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        return [Boxes(x) for x in anchors_over_all_feature_maps]