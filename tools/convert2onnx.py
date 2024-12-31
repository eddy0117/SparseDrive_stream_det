# Copyright (c) OpenMMLab. All rights reserved.
import os
import socket
import time
from os import path as osp

import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import (
    load_checkpoint,
    wrap_fp16_model,
)
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.datasets import replace_ImageToTensor, build_dataset
from mmdet.datasets import build_dataloader as build_dataloader_origin
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from tools.utils import *

MAX_CHUNK_SIZE = 5000

CLASS = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

SCORE_THRESH = 0.3

MAP_SCORE_THRESH = 0.3

# VERSION = 'v1.0-trainval'
VERSION = "v1.0-mini"

MAP_SIZE = 682

IS_SWEEP = False


set_random_seed(42)

class DummyModel(nn.Module):
    def __init__(self) -> None:
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(2, 10)
    
    def forward(self, x):
        x = torch.atan(x[..., 1])
        # x = x[..., [10, 20]].squeeze(0)
        return x

class ModelWrapper(nn.Module):
    def __init__(self, model, data) -> None:
        super(ModelWrapper, self).__init__()
        self.model = model
        self.img_metas = data["img_metas"]
        self.img = data["img"]
        self.timestamp = data["timestamp"]
        self.projection_mat = data["projection_mat"]
        self.image_wh = data["image_wh"]
        self.ego_status = data["ego_status"]
        self.gt_ego_fut_cmd = data["gt_ego_fut_cmd"]

    def forward(self, img):

        data = {
            "img_metas": [{k: torch.tensor(v) for k, v in self.img_metas.data[0][0].items()}],
            "img": img.to("cuda"),
            "timestamp": self.timestamp.to("cuda"),
            "projection_mat": self.projection_mat.to("cuda"),
            "image_wh": self.image_wh.to("cuda"),
            # DataContainer below
            "ego_status": self.ego_status,
            "gt_ego_fut_cmd": self.gt_ego_fut_cmd.data[0],
        }
        feature_maps = [torch.randn((1, 89760, 256)).to("cuda"),
                        torch.tensor([[[ 64, 176],
                                    [ 32,  88],
                                    [ 16,  44],
                                    [  8,  22]],

                                    [[ 64, 176],
                                    [ 32,  88],
                                    [ 16,  44],
                                    [  8,  22]],

                                    [[ 64, 176],
                                    [ 32,  88],
                                    [ 16,  44],
                                    [  8,  22]],

                                    [[ 64, 176],
                                    [ 32,  88],
                                    [ 16,  44],
                                    [  8,  22]],

                                    [[ 64, 176],
                                    [ 32,  88],
                                    [ 16,  44],
                                    [  8,  22]],

                                    [[ 64, 176],
                                    [ 32,  88],
                                    [ 16,  44],
                                    [  8,  22]]]).to("cuda"),
                        torch.tensor([[    0, 11264, 14080, 14784],
                                    [14960, 26224, 29040, 29744],
                                    [29920, 41184, 44000, 44704],
                                    [44880, 56144, 58960, 59664],
                                    [59840, 71104, 73920, 74624],
                                    [74800, 86064, 88880, 89584]]).to("cuda"),]
        model_outs = self.model.head(feature_maps, data)
        result = self.model.head.post_process(model_outs, data)[0]
        
        # result = self.model(return_loss=False, rescale=True, **data)[0]['img_bbox']

        for k, v in result.items():
            if isinstance(v, list):
                for it in v:
                    result[k] = torch.tensor(it).to("cuda")
            elif isinstance(v, np.ndarray):
                result[k] = torch.tensor(v).to("cuda")
        

        boxes_3d = result["boxes_3d"]
        scores_3d = result["scores_3d"]
        labels_3d = result["labels_3d"]
        cls_scores = result["cls_scores"]
        instance_ids = result["instance_ids"]
        vectors = result["vectors"]
        scores = result["scores"]
        labels = result["labels"]
        trajs_3d = result["trajs_3d"]
        trajs_score = result["trajs_score"]
        anchor_queue = result["anchor_queue"]
        period = result["period"]
        planning_score = result["planning_score"]
        planning = result["planning"]
        final_planning = result["final_planning"]
        ego_period = result["ego_period"]
        ego_anchor_queue = result["ego_anchor_queue"]

        return (
            boxes_3d,
            scores_3d,
            labels_3d,
            cls_scores,
            instance_ids,
            vectors,
            scores,
            labels,
            trajs_3d,
            trajs_score,
            anchor_queue,
            period,
            planning_score,
            planning,
            final_planning,
            ego_period,
            ego_anchor_queue,
        )


def main():
    config_path = "projects/configs/sparsedrive_small_stage2.py"
    checkpoint_path = "ckpts/sparsedrive_stage2.pth"

    lidar2img_fixed_dict = {}

    samples_per_gpu = 1

    cams = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    # socket connection setup

   

    cfg = Config.fromfile(config_path)

    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, "plugin"):
        if cfg.plugin:
            import importlib

            if hasattr(cfg, "plugin_dir"):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(config_path))[0]
        )
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.data.test.work_dir = cfg.work_dir
    print("work_dir: ", cfg.work_dir)
    # build the dataloader
    dataset = build_dataset(cfg.data.test)

    data_loader = build_dataloader_origin(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    # model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")

    # model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    # palette for visualization in segmentation tasks
    if "PALETTE" in checkpoint.get("meta", {}):
        model.PALETTE = checkpoint["meta"]["PALETTE"]

    # model = MMDataParallel(model, device_ids=[0])
    model.to("cuda")
    model.eval()

    # import onnx
    # model = onnx.load("model.onnx")
    # onnx.checker.check_model(model)
    # return
    # import onnxruntime as ort

    # ort_session = ort.InferenceSession("model.onnx")

    with torch.no_grad():
        for i, data in enumerate(data_loader):

            model_wrapper = ModelWrapper(model, data)

      
            output = model_wrapper(data['img'].data[0])
            
            

            torch.onnx.export(model_wrapper, (data['img'].data[0], ), "model.onnx", verbose=True, opset_version=14, do_constant_folding=False)
            # dummy_model = DummyModel()
            # dummy_model.to("cuda")
            # dummy_model.eval()
            # torch.onnx.export(dummy_model, (torch.zeros((1, 1, 100)).to('cuda'), ), "model.onnx", verbose=True, opset_version=14)
            break
        # output = model(return_loss=False, rescale=True, **input_data)

    

    # =============== 模型結果送到 GUI 的資料處理 ================
   


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method(
    #     "fork"
    # )  # use fork workers_per_gpu can be > 1
    main()
