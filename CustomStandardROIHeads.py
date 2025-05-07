from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.box_head import build_box_head
from CustomFastRCNNOutputLayers import CustomFastRCNNOutputLayers  # your subclass
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads
from detectron2.modeling.poolers import ROIPooler

@ROI_HEADS_REGISTRY.register()
class CustomStandardROIHeads(StandardROIHeads):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # Ensure channel counts are equal.
        in_channels = [input_shape[f].channels for f in in_features]
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Build the standard box head.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        # Build your custom predictor instead of FastRCNNOutputLayers.
        box_predictor = CustomFastRCNNOutputLayers(
            #ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
            box_head.output_shape,
            box2box_transform=Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            num_classes=cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            cls_agnostic_bbox_reg=cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            smooth_l1_beta=cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            box_reg_loss_type=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            test_score_thresh=cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            test_nms_thresh=cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            test_topk_per_image=cfg.TEST.DETECTIONS_PER_IMAGE,
            loss_weight={"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            loss_cls_weight=cfg.MODEL.ROI_BOX_HEAD.LOSS_CLS_WEIGHT,
            aperture_loss_weight=cfg.MODEL.ROI_BOX_HEAD.APERTURE_LOSS_WEIGHT,  # or set via cfg
        )
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }