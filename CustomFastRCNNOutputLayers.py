import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple, Union
from detectron2.layers import ShapeSpec, cat, batched_nms
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from detectron2.structures import Boxes, Instances
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

def l3_loss(pred, target, reduction="mean"):
    error = torch.abs(pred - target)  # absolute error
    loss = error ** 3
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
    
def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]

class CustomFastRCNNOutputLayers(FastRCNNOutputLayers):
    """
    A custom FastRCNN output layer that predicts an extra regression value "aperture"
    in addition to the usual 4 box deltas.
    """
    def __init__(self, input_shape: ShapeSpec, *, box2box_transform: Box2BoxTransform, 
                 num_classes: int, aperture_loss_weight: float = 1.0,loss_cls_weight:float=1.0, **kwargs):
        # first, call the parent so that configuration (e.g. cls_score) is set up
        super().__init__(input_shape, box2box_transform=box2box_transform, num_classes=num_classes, **kwargs)
        # original box regression dimension is 4; add 1 for aperture prediction
        self.box_reg_dim = 4
        self.total_reg_dim = self.box_reg_dim + 1
        # determine number of regression heads
        cls_agnostic_bbox_reg = kwargs.get("cls_agnostic_bbox_reg", False)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        in_channels = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.bbox_pred = nn.Linear(in_channels, num_bbox_reg_classes * self.total_reg_dim)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)
        # weight to balance the aperture loss relative to the box regression loss
        self.aperture_loss_weight = aperture_loss_weight
        self.loss_cls_weight=loss_cls_weight

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

    def losses(self, predictions, proposals):
        scores, proposal_deltas = predictions

        # Classification loss stays unchanged.
        gt_classes = cat([p.gt_classes for p in proposals], dim=0) if proposals else torch.empty(0)
        loss_cls = F.cross_entropy(scores, gt_classes, reduction="mean")

        # Determine number of regression heads used (either 1 or num_classes)
        num_bbox_reg_classes = 1 if self.bbox_pred.out_features // self.total_reg_dim == 1 else self.num_classes
        N = proposal_deltas.size(0)
        # reshape to (N, num_classes, total_reg_dim)
        pred = proposal_deltas.view(N, num_bbox_reg_classes, self.total_reg_dim)

        # Split predictions:
        # • pred_boxes: first 4 values used for box regression.
        # • pred_aperture: the extra value.
        pred_boxes = pred[:, :, :self.box_reg_dim].contiguous()
        pred_aperture = pred[:, :, self.box_reg_dim].contiguous()

        # Gather ground truth values.
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # shape (N, 4)
            gt_boxes = cat(
                [ (p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals ],
                dim=0
            )
            # Here you must have added a "gt_aperture" field to your proposals.
            gt_aperture = cat([p.gt_aperture for p in proposals], dim=0)
        else:
            proposal_boxes = gt_boxes = gt_aperture = torch.empty((0, 4), device=proposal_deltas.device)

        # Only compute regression losses for foreground proposals.
        fg_inds = (gt_classes >= 0) & (gt_classes < self.num_classes)
        if fg_inds.sum() == 0:
            loss_box_reg = proposal_deltas.sum() * 0
            loss_aperture = proposal_deltas.sum() * 0
        else:
            if pred_boxes.shape[1] > 1:
                # if class-specific, select the predictions corresponding to each gt class.
                gt_classes_clamped = gt_classes.clamp(0, num_bbox_reg_classes - 1)
                pred_boxes = pred_boxes[torch.arange(N, device=pred_boxes.device), gt_classes_clamped]
                pred_aperture = pred_aperture[torch.arange(N, device=pred_aperture.device), gt_classes_clamped]
            else:
                pred_boxes = pred_boxes[:, 0]
                pred_aperture = pred_aperture[:, 0]
            
            # Compute the box regression loss (e.g., using smooth L1)
            loss_box_reg = _dense_box_regression_loss(
                [proposal_boxes[fg_inds]],
                self.box2box_transform,
                #[pred_boxes.unsqueeze(0)[fg_inds]],
                [pred_boxes.unsqueeze(0)[:, fg_inds, :]],
                [gt_boxes[fg_inds]],
                ...,
                self.box_reg_loss_type,
                self.smooth_l1_beta,
            )
            loss_box_reg = loss_box_reg / max(gt_classes.numel(), 1.0)

            # Compute the aperture regression loss (also smooth L1)
            #loss_aperture = F.smooth_l1_loss(pred_aperture[fg_inds], gt_aperture[fg_inds].squeeze(1), reduction="sum")
            loss_aperture = F.l1_loss(pred_aperture[fg_inds], gt_aperture[fg_inds].squeeze(1), reduction="sum")
            #loss_aperture = l3_loss(pred_aperture[fg_inds], gt_aperture[fg_inds].squeeze(1), reduction="sum")
            loss_aperture = loss_aperture / max(gt_classes.numel(), 1.0)

        losses = {
            "loss_cls": loss_cls * self.loss_cls_weight,
            "loss_box_reg": loss_box_reg,
            "loss_aperture": loss_aperture * self.aperture_loss_weight,
        }
        return losses
    
    def predict_boxes_with_aperture(self, predictions, proposals):
        from detectron2.layers import cat
        scores, proposal_deltas = predictions
        # Ensure the regression outputs are a multiple of total_reg_dim (5)
        if proposal_deltas.shape[1] % self.total_reg_dim == 0:
            num_bbox_reg_classes = proposal_deltas.shape[1] // self.total_reg_dim
            # Reshape to (N, num_bbox_reg_classes, 5)
            all_deltas = proposal_deltas.view(-1, num_bbox_reg_classes, self.total_reg_dim)
            # Split into box deltas (first 4) and aperture (fifth)
            box_deltas = all_deltas[:, :, :4].reshape(all_deltas.size(0), -1)
            aperture_values = all_deltas[:, :, 4]  # shape: (N, num_bbox_reg_classes)
        else:
            raise ValueError("Regression predictions do not align with expected dimensions")
        # Gather proposals from RPN
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        # Decode boxes using the first 4 deltas
        pred_boxes = self.box2box_transform.apply_deltas(box_deltas, proposal_boxes)
        # Split the predictions by image
        boxes_split = pred_boxes.split(num_prop_per_image)
        aperture_split = aperture_values.split(num_prop_per_image)
        return boxes_split, aperture_split
    
    def inference(self, predictions, proposals):
        # Get decoded boxes and aperture values from your custom method.
        boxes_split, aperture_split = self.predict_boxes_with_aperture(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [p.image_size for p in proposals]
        # Use the standard fast_rcnn_inference to get final instances, which returns (instances, kept_indices)
        pred_instances, kept_indices = fast_rcnn_inference(
            boxes_split, scores, image_shapes,
            self.test_score_thresh, self.test_nms_thresh, self.test_topk_per_image
        )
        # For each image, attach the aperture values corresponding to the kept proposals.
        # (Make sure to split aperture_split according to each image's kept indices.)
        for i, inst in enumerate(pred_instances):
            # kept_indices[i] contains indices of the proposals that survived NMS for image i.
            inst.aperture = aperture_split[i][kept_indices[i]]
        return pred_instances, kept_indices
    