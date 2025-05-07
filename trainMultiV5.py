import torch.optim as optim
from detectron2.engine import DefaultTrainer, hooks
import matplotlib.pyplot as plt
from detectron2.engine.hooks import HookBase
from detectron2.utils.events import get_event_storage
from detectron2.utils import comm  # Add this import
import logging
import random
import numpy as np
import argparse
from detectron2.evaluation import inference_on_dataset
from CustomMatchingEvaluator import CustomMatchingEvaluator
logger = logging.getLogger(__name__)

class PeriodicEvaluator(HookBase):
    def __init__(self, cfg, eval_period, dataset_name_val, dataset_name_train, evaluator, mapper, extra_iters=None):
        """
        Args:
            cfg (CfgNode): Configuration object; will be cloned inside the hook.
            eval_period (int): Evaluation frequency (in iterations).
            dataset_name_val (str): The name of the validation dataset (as registered) to evaluate.
            dataset_name_train (str): The name of the training dataset (as registered) to evaluate.
            evaluator (DatasetEvaluator): An instance of your custom evaluator.
            mapper (callable): The dataset mapper used in building the loader.
        """
        self.cfg = cfg.clone()
        self._eval_period = eval_period
        self.extra_iters = set(extra_iters or [])
        self.dataset_name_val = dataset_name_val
        self.dataset_name_train = dataset_name_train
        self.evaluator = evaluator  
        self.mapper = mapper

        # Lists to store evaluation metrics over time:
        self.eval_iters = []
        self.f1_val = []
        self.f1_train = []
        self.eval_results = []  # Add this to store results

    def _do_evaluation(self):

        model = self.trainer.model
        model.eval()  # set to evaluation mode

        # --- Evaluate Validation Data ---
        data_loader_val = build_detection_test_loader(self.cfg, self.dataset_name_val, self.mapper)
        logger.info(f"Running evaluation on validation dataset: {self.dataset_name_val} ...")
        # Reset evaluator before running inference
        self.evaluator.reset()
        results_val = inference_on_dataset(model, data_loader_val, self.evaluator)
        logger.info(f"Validation results:\n{results_val}")
        # Extract detection and aperture metrics for validation:
        det_metrics_val = results_val.get('detection_metrics', {})
        aper_metrics_val = results_val.get('aperture_evaluation', {})
        precision_val = det_metrics_val.get('precision', 0)
        recall_val = det_metrics_val.get('recall', 0)
        mean_abs_error_val = aper_metrics_val.get('mean_absolute_error', 0)
        num_matches_val = aper_metrics_val.get('num_matches', 0)
        tp_val = det_metrics_val.get('true_positives', 0)
        fp_val = det_metrics_val.get('false_positives', 0)
        fn_val = det_metrics_val.get('false_negatives', 0)
        # Compute F1 for validation:
        if (precision_val + recall_val) > 0:
            f1_val = 2 * precision_val * recall_val / (precision_val + recall_val)
        else:
            f1_val = 0.0

        # --- Evaluate Training Data ---
        data_loader_train = build_detection_test_loader(self.cfg, self.dataset_name_train, self.mapper)
        logger.info(f"Running evaluation on training dataset: {self.dataset_name_train} ...")
        self.evaluator.reset()
        results_train = inference_on_dataset(model, data_loader_train, self.evaluator)
        logger.info(f"Training results:\n{results_train}")
        det_metrics_train = results_train.get('detection_metrics', {})
        aper_metrics_train = results_train.get('aperture_evaluation', {})
        precision_train = det_metrics_train.get('precision', 0)
        recall_train = det_metrics_train.get('recall', 0)
        mean_abs_error_train = aper_metrics_train.get('mean_absolute_error', 0)
        num_matches_train = aper_metrics_train.get('num_matches', 0)
        tp_train = det_metrics_train.get('true_positives', 0)
        fp_train = det_metrics_train.get('false_positives', 0)
        fn_train = det_metrics_train.get('false_negatives', 0)
        if (precision_train + recall_train) > 0:
            f1_train = 2 * precision_train * recall_train / (precision_train + recall_train)
        else:
            f1_train = 0.0

        # --- Store Evaluation Data ---
        cur_iter = self.trainer.iter
        self.eval_iters.append(cur_iter)
        self.f1_val.append(f1_val)
        self.f1_train.append(f1_train)
        # Also log scalar values for later visualization (optional)
        self.trainer.storage.put_scalar(f"{self.dataset_name_val}/AP", precision_val)
        self.trainer.storage.put_scalar(f"{self.dataset_name_train}/AP", precision_train)

        # --- Store results for txt output ---
        eval_result = {
            "iteration": cur_iter,
            "f1_val": f1_val,
            "precision_val": precision_val,
            "recall_val": recall_val,
            "mean_abs_error_val": mean_abs_error_val,
            "tp_val": tp_val,
            "fp_val": fp_val,
            "fn_val": fn_val,
            "f1_train": f1_train,
            "precision_train": precision_train,
            "recall_train": recall_train,
            "mean_abs_error_train": mean_abs_error_train,
            "tp_train": tp_train,
            "fp_train": fp_train,
            "fn_train": fn_train,
        }
        self.eval_results.append(eval_result)

        # --- Generate Plot ---
        fig, ax = plt.subplots(figsize=(10, 6))
        # Plot F1 curves
        ax.plot(self.eval_iters, self.f1_val, marker='o', label="Validation F1")
        ax.plot(self.eval_iters, self.f1_train, marker='o', label="Training F1")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("F1 Score")
        ax.set_title("F1 Score Over Iterations")
        ax.legend()
        ax.grid(True)

        # Create text box for validation metrics:
        metrics_text_val = (
            f"[Validation]\n"
            f"Precision: {precision_val:.2f}\n"
            f"Recall: {recall_val:.2f}\n"
            f"F1: {f1_val:.2f}\n"
            f"TP: {tp_val}, FP: {fp_val}, FN: {fn_val}\n"
            f"Mean Abs Error: {mean_abs_error_val:.2f}\n"
            f"Matches: {num_matches_val}"
        )

        # And a text box for training metrics:
        metrics_text_train = (
            f"[Training]\n"
            f"Precision: {precision_train:.2f}\n"
            f"Recall: {recall_train:.2f}\n"
            f"F1: {f1_train:.2f}\n"
            f"TP: {tp_train}, FP: {fp_train}, FN: {fn_train}\n"
            f"Mean Abs Error: {mean_abs_error_train:.2f}\n"
            f"Matches: {num_matches_train}"
        )

        """         ax = plt.gca()
        ax.text(0.05, 0.95, metrics_text_val,
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.65, 0.95, metrics_text_train,
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)) """
        fig.subplots_adjust(right=0.75)  # make space on the right
        fig.text(0.78, 0.75, metrics_text_val, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        fig.text(0.78, 0.3, metrics_text_train, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # Save the plot to a fixed file (this overwrites previous one)
        if comm.is_main_process():
            plot_path = os.path.join(self.cfg.OUTPUT_DIR, "evaluation_plot.png")
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
            plt.close()

            txt_path = os.path.join(self.cfg.OUTPUT_DIR, "evaluation_results.txt")
            with open(txt_path, "w") as f:
                for res in self.eval_results:
                    f.write(f"iteration:{res['iteration']}\n")
                    f.write(
                        f"f1_val:{res['f1_val']:.4f}, precision_val:{res['precision_val']:.4f}, recall_val:{res['recall_val']:.4f}, "
                        f"mean_abs_error_val:{res['mean_abs_error_val']:.4f}, TP:{res['tp_val']}, FP:{res['fp_val']}, FN:{res['fn_val']}\n"
                    )
                    f.write(
                        f"f1_train:{res['f1_train']:.4f}, precision_train:{res['precision_train']:.4f}, recall_train:{res['recall_train']:.4f}, "
                        f"mean_abs_error_train:{res['mean_abs_error_train']:.4f}, TP:{res['tp_train']}, FP:{res['fp_train']}, FN:{res['fn_train']}\n"
                    )
                    f.write("--------------------------------------------------------------------------------------------\n")

        # Switch back to train mode
        model.train()
    
    def after_step(self):
        it = self.trainer.iter

        """         # 1) Only let the rank-0 process do the evaluation
        if not comm.is_main_process():
            return """

        # 2) Fire if (a) you hit the periodic boundary or (b) you're in extra_iters
        do_periodic = (it > 0 and it % self._eval_period == 0)
        do_extra    = (it in self.extra_iters)
        if do_periodic or do_extra:
            # synchronize everyone up to here (mostly a no-op on non-rank-0)
            comm.synchronize()
            self._do_evaluation()
            comm.synchronize()

# Add this function after your imports
def set_seed(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For complete determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_unique_output_dir(base_dir="output"):
    """
    Returns a unique directory name by appending an integer suffix if necessary.
    For example, if "output" exists, it returns "output2", then "output3", etc.
    """
    output_dir = base_dir
    counter = 1
    while os.path.exists(output_dir):
        output_dir = f"{base_dir}{counter}"
        counter += 1
    return output_dir

class GradientPlotter(HookBase):
    def __init__(self, plot_period=500, output_file="gradients.png"):
        self.plot_period = plot_period
        self.output_file = output_file
        self.iterations = []
        self.grad_aperture = []
        self.grad_cls = []
        self.grad_bbox = []
        self.grad_rpn_cls = []
        self.grad_rpn_loc = []

    def after_step(self):
        # Only run on main process
        if not comm.is_main_process():
            return
        iteration = self.trainer.iter
        if iteration % self.plot_period == 0:
            self.iterations.append(iteration)
            model = self.trainer.model

            # --- From the box predictor ---
            actual_model = model.module if hasattr(model, "module") else model
            predictor = actual_model.roi_heads.box_predictor
            # Ensure gradients exist (assumes they were computed during loss.backward())
            if predictor.bbox_pred.weight.grad is not None:
                weight_grad = predictor.bbox_pred.weight.grad
                total_reg_dim = predictor.total_reg_dim  # should be 5 for your custom predictor
                num_bbox_reg_classes = weight_grad.shape[0] // total_reg_dim
                reshaped_weight_grad = weight_grad.view(num_bbox_reg_classes, total_reg_dim, -1)
                # Aperture gradients: index 4 (0-indexed) for each head
                aperture_weight_grad = reshaped_weight_grad[:, 4, :]
                grad_ap = aperture_weight_grad.norm(dim=1).mean().item()  # average over heads
            else:
                grad_ap = 0.0

            # Classification gradients: from cls_score layer
            if predictor.cls_score.weight.grad is not None:
                cls_weight_grad = predictor.cls_score.weight.grad
                grad_cls = cls_weight_grad.norm(dim=1).mean().item()
            else:
                grad_cls = 0.0

            # Bounding box gradients: indices 0-3 in bbox_pred
            if predictor.bbox_pred.weight.grad is not None:
                bbox_weight_grad = reshaped_weight_grad[:, :4, :]
                grad_bbox = bbox_weight_grad.norm(dim=(1,2)).mean().item()
            else:
                grad_bbox = 0.0

            # --- From the RPN head ---
            rpn_head = actual_model.proposal_generator.rpn_head
            if rpn_head.objectness_logits.weight.grad is not None:
                rpn_cls_grad = rpn_head.objectness_logits.weight.grad.norm().item()
            else:
                rpn_cls_grad = 0.0
            if rpn_head.anchor_deltas.weight.grad is not None:
                rpn_loc_grad = rpn_head.anchor_deltas.weight.grad.norm().item()
            else:
                rpn_loc_grad = 0.0

            self.grad_aperture.append(grad_ap)
            self.grad_cls.append(grad_cls)
            self.grad_bbox.append(grad_bbox)
            self.grad_rpn_cls.append(rpn_cls_grad)
            self.grad_rpn_loc.append(rpn_loc_grad)

            # Plot the gradient norms
            plt.figure(figsize=(10, 6))
            plt.plot(self.iterations, self.grad_aperture, label="Grad Aperture")
            plt.plot(self.iterations, self.grad_cls, label="Grad Classification")
            plt.plot(self.iterations, self.grad_bbox, label="Grad BBox")
            plt.plot(self.iterations, self.grad_rpn_cls, label="Grad RPN Cls")
            plt.plot(self.iterations, self.grad_rpn_loc, label="Grad RPN Loc")
            plt.xlabel("Iteration")
            plt.ylabel("Gradient Norm")
            plt.title("Gradient Norms Over Training")
            plt.legend()
            plt.grid(True)
            # Save the plot in the output directory:
            out_path = os.path.join(self.trainer.cfg.OUTPUT_DIR, self.output_file)
            plt.savefig(out_path)
            plt.close()

class LossPlotter(HookBase):
    def __init__(self, plot_period=500, output_file="loss.png", window_size=20):
        """
        Args:
            plot_period (int): How often (in iterations) to update the plot.
            output_file (str): The file path where the plot PNG is saved.
            window_size (int): Window size for moving average of loss values.
        """
        self.plot_period = plot_period
        self.output_file = output_file
        self.window_size = window_size
        self.iterations = []
        self.loss_cls = []
        self.loss_box_reg = []
        self.loss_aperture = []
        self.loss_rpn_cls = []
        self.loss_rpn_loc = []
        self.total_loss = []

    def after_step(self):
        if not comm.is_main_process():
            return
            
        iteration = self.trainer.iter
        if iteration % self.plot_period == 0:
            storage = get_event_storage()
            
            self.iterations.append(iteration)
            try:
                # Get smoothed values with specified window size
                loss_dict = {
                    "loss_cls": storage.history("loss_cls").avg(self.window_size),
                    "loss_box_reg": storage.history("loss_box_reg").avg(self.window_size),
                    "loss_aperture": storage.history("loss_aperture").avg(self.window_size),
                    "loss_rpn_cls": storage.history("loss_rpn_cls").avg(self.window_size),
                    "loss_rpn_loc": storage.history("loss_rpn_loc").avg(self.window_size)
                }
                
                # Append values to lists
                self.loss_cls.append(float(loss_dict["loss_cls"]))
                self.loss_box_reg.append(float(loss_dict["loss_box_reg"]))
                self.loss_aperture.append(float(loss_dict["loss_aperture"]))
                self.loss_rpn_cls.append(float(loss_dict["loss_rpn_cls"]))
                self.loss_rpn_loc.append(float(loss_dict["loss_rpn_loc"]))
                
                # Calculate total loss
                total_loss = sum(float(val) for val in loss_dict.values())
                self.total_loss.append(total_loss)

                # Create plot
                plt.figure(figsize=(10, 6))
                plt.plot(self.iterations, self.loss_cls, label="loss_cls")
                plt.plot(self.iterations, self.loss_box_reg, label="loss_box_reg")
                plt.plot(self.iterations, self.loss_aperture, label="loss_aperture")
                plt.plot(self.iterations, self.loss_rpn_cls, label="loss_rpn_cls")
                plt.plot(self.iterations, self.loss_rpn_loc, label="loss_rpn_loc")
                plt.plot(self.iterations, self.total_loss, label="total_loss", 
                        linewidth=2, color="black")
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                plt.title("Training Losses")
                plt.legend()
                plt.grid(True)
                out_path = os.path.join(self.trainer.cfg.OUTPUT_DIR, self.output_file)
                plt.savefig(out_path)
                plt.close()
                
            except KeyError as e:
                logger.warning(f"Missing metric in storage: {e}")
                return
            
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_optimizer(cls, cfg, model):
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            # Optionally adjust for biases (if desired)
            if "bias" in key:
                lr = lr * cfg.SOLVER.BIAS_LR_FACTOR
                # If WEIGHT_DECAY_BIAS is defined, use it, otherwise use 0.
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS if cfg.SOLVER.WEIGHT_DECAY_BIAS is not None else 0.0
            params.append({"params": [value], "lr": lr, "weight_decay": weight_decay})
        optimizer = optim.AdamW(params, lr=cfg.SOLVER.BASE_LR)
        return optimizer

    @classmethod
    def build_train_loader(cls, cfg):
        from detectron2.data import build_detection_train_loader
        return build_detection_train_loader(cfg, mapper=radar_mapper)

    def build_hooks(self):
        hooks_list = super().build_hooks()
        # Insert the loss plotter hook before the last hook (often the periodic checkpointer)
        hooks_list.insert(-1, LossPlotter(
        plot_period=100,
        output_file="loss.png", 
        #output_file=os.path.join(self.cfg.OUTPUT_DIR, "loss.png"),
        window_size=20  # Adjust this value as needed
        ))
        hooks_list.insert(-1, GradientPlotter(
        plot_period=100,
        output_file="gradients.png" 
        #output_file=os.path.join(self.cfg.OUTPUT_DIR, "gradients.png")
        ))

        evaluator = CustomMatchingEvaluator(iou_threshold=0.15)


        hooks_list.insert(-1, PeriodicEvaluator(
            cfg=self.cfg,
            eval_period=500,
            dataset_name_val="RADIal_COCO-style_val",
            dataset_name_train="RADIal_COCO-style_train",
            evaluator=evaluator,
            mapper=radar_mapper,
            extra_iters=None# To add extra evaluations just add a list with the iteration where you want to evaluation to be done [5000,5050]
        ))

        hooks_list.insert(-1, hooks.PeriodicCheckpointer(self.checkpointer, 5000, max_iter=self.iter))

        return hooks_list

#!/usr/bin/env python
"""
train.py – A training script for your custom radar detection model using Detectron2.
"""

import os
import copy
import numpy as np
import torch
import logging
from collections import OrderedDict

from detectron2.engine import DefaultTrainer, launch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets import load_coco_json
import detectron2.data.detection_utils as utils
from detectron2.structures import Instances, Boxes, BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Import your custom modules
from custom_anchorGen2 import CustomAnchorGenerator
from custom_backbone import CustomResNetBackbone
from CustomFastRCNNOutputLayers import CustomFastRCNNOutputLayers
from CustomStandardROIHeads import CustomStandardROIHeads

# ---- Custom Mapper (as before) ----
def radar_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    radar = np.load(dataset_dict["file_name"], allow_pickle=True)
    radar = np.concatenate([radar.real, radar.imag], axis=2)  # (H, W, C)
    radar = torch.as_tensor(radar.astype("float32")).permute(2, 0, 1)  # (C, H, W)
    PIXEL_MEAN = torch.tensor([
        -0.0026244, -0.21335,  0.018789, -1.4427, -0.37618, 1.3594,
        -0.22987,  0.12244,  1.7359, -0.65345,  0.37976, 5.5521,
         0.77462, -1.5589, -0.72473, 1.5182, -0.37189, -0.088332,
        -0.16194, 1.0984,  0.99929, -1.0495, 1.9972,  0.92869,
         1.8991, -0.23772, 2.0,  0.77737, 1.3239, 1.1817,
        -0.69696, 0.44288
    ], dtype=torch.float32)
    PIXEL_STD = torch.tensor([
        20775.3809, 23085.5, 23017.6387, 14548.6357, 32133.5547, 28838.8047,
        27195.8945, 33103.7148, 32181.5273, 35022.1797, 31259.1895, 36684.6133,
        33552.9258, 25958.7539, 29532.623, 32646.8984, 20728.332, 23160.8828,
        23069.0449, 14915.9053, 32149.6172, 28958.584, 27210.8652, 33005.6602,
        31905.9336, 35124.918, 31259.1895, 31086.0273, 33628.5352, 25950.2363,
        29445.2598, 32885.7422
    ], dtype=torch.float32)
    radar = (radar - PIXEL_MEAN[:, None, None]) / PIXEL_STD[:, None, None]
    dataset_dict["image"] = radar

    if "annotations" in dataset_dict:
        annos = dataset_dict.pop("annotations")
        for anno in annos:
            if "aperture" in anno:
                anno["gt_aperture"] = anno["aperture"]
        annos = [utils.transform_instance_annotations(annotation, [], radar.shape[1:]) for annotation in annos]
        instances = utils.annotations_to_instances(annos, radar.shape[1:])
        gt_apertures = [anno.get("gt_aperture") for anno in annos]
        if len(gt_apertures) > 0:
            instances.gt_aperture = torch.tensor(gt_apertures, dtype=torch.float32).unsqueeze(1)
        dataset_dict["instances"] = instances
    return dataset_dict

# ---- Dataset Registration (as before) ----
def register_datasets():
    train_json = "/home/eorozco/projects/radarObjDet/cocoRadialTrain.json"
    val_json = "/home/eorozco/projects/radarObjDet/cocoRadialVal.json"
    image_root = "/volumes/8TB_volume1/eorozco/RADIal/RADIal/radar_FFT"
    train_name = "RADIal_COCO-style_train"
    val_name = "RADIal_COCO-style_val"

    DatasetCatalog.register(
        train_name,
        lambda: load_coco_json(
            json_file=train_json,
            image_root=image_root,
            dataset_name=train_name,
            extra_annotation_keys=["aperture"]
        )
    )
    MetadataCatalog.get(train_name).set(thing_classes=["car"], evaluator_type="coco")

    DatasetCatalog.register(
        val_name,
        lambda: load_coco_json(
            json_file=val_json,
            image_root=image_root,
            dataset_name=val_name,
            extra_annotation_keys=["aperture"]
        )
    )
    MetadataCatalog.get(val_name).set(thing_classes=["car"], evaluator_type="coco")

# ---- Configuration Setup (modify paths and hyperparameters as needed) ----
def setup_cfg():
    cfg = get_cfg()
    cfg.SEED = 182
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    cfg.MODEL.DEVICE = "cuda"
    cfg.DATALOADER.NUM_WORKERS = 2
    #cfg.MODEL.WEIGHTS = ""  # Train from scratch
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.DATASETS.TRAIN = ("RADIal_COCO-style_train",)
    cfg.DATASETS.TEST = ("RADIal_COCO-style_val",)

    # Backbone & FPN settings
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 192
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"] #removed 1 block
    cfg.MODEL.RESNETS.NORM = ""
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"] #removed 1 block

    # Anchor generator settings
    cfg.MODEL.ANCHOR_GENERATOR.STRIDES = [1, 2, 4, 8]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[71], [71], [71], [71]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.078125]]
    cfg.MODEL.ANCHOR_GENERATOR.OFFSET = 0.5
    cfg.MODEL.ANCHOR_GENERATOR.STD_BEHAVIOR = True
    cfg.MODEL.ANCHOR_GENERATOR.NAME = "CustomAnchorGenerator"

    # Custom ROI Heads & Box Predictor settings
    cfg.MODEL.ROI_HEADS.NAME = "CustomStandardROIHeads"
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p0", "p1","p2", "p3"] #removed 1 block
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.35
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.3
    cfg.MODEL.RPN.IN_FEATURES = ["p0", "p1", "p2","p3"] #removed 1 blockss
    cfg.MODEL.FPN.OUT_CHANNELS = 256
    cfg.MODEL.FPN.NORM = ""
    cfg.MODEL.FPN.FUSE_TYPE = "sum"
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
   
   #set custom backbone
    cfg.MODEL.BACKBONE.NAME = "CustomResNetBackbone"

    # Box Regression settings (note: your custom predictor outputs 5 values)
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (1.0, 5.0, 1.0, 5.0)
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "smooth_l1"
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_BOX_HEAD.APERTURE_LOSS_WEIGHT = 22
    cfg.MODEL.ROI_BOX_HEAD.LOSS_CLS_WEIGHT=2
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True

    # Pixel normalization for 32-channel radar data
    cfg.MODEL.PIXEL_MEAN = [
        -0.0026244, -0.21335, 0.018789, -1.4427, -0.37618, 1.3594,
        -0.22987, 0.12244, 1.7359, -0.65345, 0.37976, 5.5521,
         0.77462, -1.5589, -0.72473, 1.5182, -0.37189, -0.088332,
        -0.16194, 1.0984, 0.99929, -1.0495, 1.9972, 0.92869,
         1.8991, -0.23772, 2.0, 0.77737, 1.3239, 1.1817,
        -0.69696, 0.44288
    ]
    cfg.MODEL.PIXEL_STD = [
        20775.3809, 23085.5, 23017.6387, 14548.6357, 32133.5547, 28838.8047,
        27195.8945, 33103.7148, 32181.5273, 35022.1797, 31259.1895, 36684.6133,
        33552.9258, 25958.7539, 29532.623, 32646.8984, 20728.332, 23160.8828,
        23069.0449, 14915.9053, 32149.6172, 28958.584, 27210.8652, 33005.6602,
        31905.9336, 35124.918, 31259.1895, 31086.0273, 33628.5352, 25950.2363,
        29445.2598, 32885.7422
    ]

    # Solver settings
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 8010
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.STEPS = (8000, 8005)
    cfg.TEST.DETECTIONS_PER_IMAGE = 10
    cfg.OUTPUT_DIR = get_unique_output_dir("output")
    #os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg

# ---- Main Function ----
def main():
    set_seed(182)
    cfg = setup_cfg()
        # Create output dir only in main process
    if comm.is_main_process():
        cfg.OUTPUT_DIR = get_unique_output_dir(cfg.OUTPUT_DIR)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(cfg.OUTPUT_DIR, "config.txt"), "w") as f:
            f.write(cfg.dump())
    
    # Synchronize all processes to ensure output dir is created
    comm.synchronize()
    register_datasets()
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-url", type=str, default="auto")
    args = parser.parse_args()
    # Launch multi-GPU training if desired. For example, to use 2 GPUs:
    launch(main, num_gpus_per_machine=4, num_machines=1, machine_rank=0, dist_url=args.dist_url)
