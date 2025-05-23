from inspect import signature

import torch

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from .grid_mask import GridMask
from torch.cuda.amp.autocast_mode import autocast

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["SparseDrive"]

# import torch_tensorrt
from .my_models.fpn import FPN

@DETECTORS.register_module()
class SparseDrive(BaseDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
    ):
        super(SparseDrive, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            backbone.pretrained = pretrained
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            # self.img_neck = build_neck(img_neck)
            img_neck.pop("type")
            self.img_neck = FPN(**img_neck)
        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            ) 
        import copy
        self.img_backbone.eval()
        self.inference_stream = torch.cuda.Stream()
        # self.img_backbone_ori = copy.deepcopy(self.img_backbone)
        

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)

        # with torch.cuda.stream(self.inference_stream):
        #     img = img.half().cuda(non_blocking=True)
        feature_maps = self.img_backbone(img)
        # self.inference_stream.synchronize()

        if self.img_neck is not None:
            feature_maps = list(self.img_neck(*feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)
        # with autocast(enabled=True, dtype=torch.bfloat16):
        model_outs = self.head(feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):

        # MODIFIED 移除不必要的key (for motion planning)

        # data.pop('gt_ego_fut_cmd')
        # data.pop('return_loss')
        # data.pop('rescale')
        
        import time
        # t0 = time.time()
        feature_maps = self.extract_feat(img)
        # ext_time = round((time.time() - t0) * 1000, 2)
        # t0 = time.time()
        model_outs = self.head(feature_maps, data)
        # head_time = round((time.time() - t0) * 1000, 2)
        # t0 = time.time()
        results = self.head.post_process(model_outs, data)
        # post_time = round((time.time() - t0) * 1000, 2)
        output = [{"img_bbox": result} for result in results]
        # print(f"ext_time: {ext_time}ms, head_time: {head_time}ms, post_time: {post_time}ms")
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
