# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
import os
from os import path as osp

import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)

from mmdet.apis import single_gpu_test, multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor, build_dataset
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmdet.models import build_detector
import sys
sys.path.insert(0, '/home/eddy/Programs/SparseDrive_stream_det')

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.apis.test import custom_multi_gpu_test

from tools.utils import *
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

import time
import base64
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
import numpy as np
import cv2
import torch
import os
import socket
import json

MAX_CHUNK_SIZE = 5000

CLASS = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

SCORE_THRESH = 0.3

MAP_SCORE_THRESH = 0.3

# VERSION = 'v1.0-trainval'
VERSION = 'v1.0-mini'


IS_SWEEP = False

# 是否將模型輸出透過TCP發送
IS_SERVER = False
IS_MAP = True
IS_TRAJ = False

set_random_seed(42)

def main():

    config_path = 'projects/configs/sparsedrive_small_stage2.py'
    checkpoint_path = 'ckpts/sparsedrive_stage2.pth'

    # config_path = 'projects/configs/sparsedrive_small_stage1.py'
    # checkpoint_path = 'ckpts/our_train_stage1.pth'
    # checkpoint_path = 'ckpts/our_train_stage1_bit.pth'
    # checkpoint_path = 'ckpts/sparsedrive_stage1.pth'

    lidar2img_fixed_dict = {}
    
   
    samples_per_gpu = 1

    aug_config = {'resize': 0.44, 'resize_dims': (704, 396), 'crop': (0, 140, 704, 396), 'flip': False, 'rotate': 0, 'rotate_3d': 0}

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    scene_list = ['scene-0103', 'scene-0916']
    # scene_list = ['scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016']
    
    
    # scene_list = ['scene-1094', 'scene-1100']
    # scene_list = ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100']

    # scene_list = ['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
    #  'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',]
    cams = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    
    sd_rec_c_dict = {}

    bev_idx = 0
    infer_time_arr = []
    # socket connection setup

    if IS_SERVER:

        print('connecting to server')
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 65432))
        print('finish connecting')

    cfg = Config.fromfile(config_path)
    


    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated

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



    model = MMDataParallel(model, device_ids=[0])


    model.eval()

    nusc = NuScenes(version=VERSION, dataroot='data/nuscenes/', verbose=True)
    
    # read can bus data
    nusc_can = NuScenesCanBus(dataroot='data')
    
    
    
    for scene_name in scene_list:

        idx = 0
        
        can_list = nusc_can.get_messages(scene_name ,'vehicle_monitor')

        scene = next(scene for scene in nusc.scene if scene['name'] == scene_name) 
        # scene_token = scene['token']


        sample = nusc.get('sample', scene['first_sample_token'])

        

        while sample['next'] != '':

           

            t0 = time.time()
            img_load_time_arr = []
            intrinsic_arr = []
            extrinsic_arr = []
            lidar2img_arr = []
            filename_arr = []
            img_arr = []
            img_ori_arr = []
            

            # loop start
            
            if idx == 0 or not IS_SWEEP:

                sd_rec_l = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

            elif IS_SWEEP:

                sd_rec_l = nusc.get('sample_data', sd_rec_l['next'])

            cs_record_l = nusc.get('calibrated_sensor', sd_rec_l['calibrated_sensor_token'])

            pose_record_l = nusc.get('ego_pose', sd_rec_l['ego_pose_token'])

            timestamp = sd_rec_l['timestamp'] / 1e6

            # Optimize tensor conversion using torch.as_tensor with unified device
            device = torch.device("cuda")
            e2g_rotation = torch.as_tensor(Quaternion(pose_record_l['rotation']).rotation_matrix, device=device).detach()
            e2g_translation = torch.as_tensor(pose_record_l['translation'], device=device).detach()
            l2e_rotation = torch.as_tensor(Quaternion(cs_record_l['rotation']).rotation_matrix, device=device).detach()
            l2e_translation = torch.as_tensor(cs_record_l['translation'], device=device).detach()
            
            e2g_matrix = convert_egopose_to_matrix_numpy(e2g_rotation, e2g_translation)
            l2e_matrix = convert_egopose_to_matrix_numpy(l2e_rotation, l2e_translation)

            ego_pose = e2g_matrix @ l2e_matrix
            ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
            
            # ego_status

            ego_status = get_ego_status(nusc, nusc_can, sample)

            # lidar2img & intrinsic & extrinsic
            for cam in cams:
                
                if idx == 0 or not IS_SWEEP:
                
                    cam_token = sample['data'][cam]
                    # cam_token_dict[cam] = sample['data'][cam]

                    # 使用 token 获取 sample_data 记录
                    sd_rec_c = nusc.get('sample_data', cam_token)
                    sd_rec_c_dict[cam] = sd_rec_c

                elif IS_SWEEP:

                    sd_rec_c = nusc.get('sample_data', sd_rec_c_dict[cam]['next'])
                    sd_rec_c_dict[cam] = sd_rec_c

                # 获取摄像头的校准数据
                cs_record_c = nusc.get('calibrated_sensor', sd_rec_c['calibrated_sensor_token'])

                pose_record_c = nusc.get('ego_pose', sd_rec_c['ego_pose_token'])
                

                
                # lidar
                # l2e_t = np.array(cs_record_l['translation'])
                # e2g_t = np.array(pose_record_l['translation'])
                # l2e_r = np.array(cs_record_l['rotation'])
                # e2g_r = np.array(pose_record_l['rotation'])

                # cam
                # cam to ego(car center)
                c2e_t_s = torch.as_tensor(cs_record_c['translation'], device=device).detach()
                c2e_r_s = np.array(cs_record_c['rotation'])

                # cam to global
                e2g_t_s = torch.as_tensor(pose_record_c['translation'], device=device).detach()
                e2g_r_s = np.array(pose_record_c['rotation'])

                # l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                # e2g_r_mat = Quaternion(e2g_r).rotation_matrix
                l2e_r_mat = l2e_rotation
                e2g_r_mat = e2g_rotation
                l2e_t = l2e_translation
                e2g_t = e2g_translation

                c2e_r_s_mat = torch.as_tensor(Quaternion(c2e_r_s).rotation_matrix, device=device).float().detach()
                e2g_r_s_mat = torch.as_tensor(Quaternion(e2g_r_s).rotation_matrix, device=device).float().detach()
                
                R = (c2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
                        torch.linalg.inv(e2g_r_mat).float().T @ torch.linalg.inv(l2e_r_mat).float().T)
                T = (c2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
                    torch.linalg.inv(e2g_r_mat).float().T @ torch.linalg.inv(l2e_r_mat).float().T)
                T -= e2g_t @ (torch.linalg.inv(e2g_r_mat).float().T @ torch.linalg.inv(l2e_r_mat).float().T
                                ) + l2e_t @ torch.linalg.inv(l2e_r_mat).float().T

                cam2lidar_r = R.T
                cam2lidar_t = T

                cam2lidar_rt = convert_egopose_to_matrix_numpy(cam2lidar_r, cam2lidar_t)
                lidar2cam_rt = invert_matrix_egopose_numpy(cam2lidar_rt)

                intrinsic = torch.as_tensor(cs_record_c['camera_intrinsic'], device=device).detach()

                viewpad = torch.eye(4).cuda()
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

                # lidar2img
                lidar2img = (viewpad @ lidar2cam_rt)

                intrinsic = viewpad
                extrinsic = lidar2cam_rt
                
                
                # LoadImage
                t1 = time.time()
                img_name = 'data/nuscenes/' + sd_rec_c['filename']
                img_ori = mmcv.imread(img_name, 'unchanged')
                img_ori_arr.append(img_ori)
                img_load_time_arr.append(round((time.time() - t1) * 1000, 2))
                # ResizeCropFlipRotImage class call
                
                # t0 = time.time()
                img, mat = img_transform(
                                img_ori,
                                aug_config
                            )
                # intrinsic[:3, :3] = ida_mat @ intrinsic[:3, :3]
                
                # lidar2img = intrinsic @ extrinsic
                
                lidar2img = torch.as_tensor(mat, device=device).float().detach() @ lidar2img

                #===================================
    
                
                # NormalizeImage
                
                img = mmcv.imnormalize(img, mean, std, False)

                intrinsic_arr.append(intrinsic)
                extrinsic_arr.append(extrinsic)
                lidar2img_arr.append(lidar2img)
                filename_arr.append('data/nuscenes/' + sd_rec_c['filename'])
                img_arr.append(torch.as_tensor(img.transpose(2, 0, 1), device=device).float())
                # print('preprocess time', time.time() - t0)

            img_arr = torch.stack(img_arr)
            projection_mat = torch.stack(lidar2img_arr)
            print('total image loading time', np.sum(img_load_time_arr), 'ms')
            img_metas = {'T_global' : ego_pose,
                    'T_global_inv' : ego_pose_inv,
                    'timestamp' : timestamp}
        

            input_data = {'img_metas': [[img_metas]],                      
                        'img': [img_arr.unsqueeze(0)],
                        'timestamp' : torch.Tensor([timestamp]).to('cuda'),
                        'projection_mat' : [projection_mat.unsqueeze(0)],
                        'image_wh' : torch.Tensor([[[704, 256] for _ in range(6)]]).to('cuda'),
                        'ego_status' : torch.Tensor([np.array(ego_status)]).to('cuda'),
                        # 'ego_status' : torch.Tensor([np.array([0] * 10)]).to('cuda'),
                        'gt_ego_fut_cmd' : torch.Tensor([np.array([0, 0, 1])]).to('cuda')
                        }
            
            prepro_time = round((time.time() - t0) * 1000, 2)
            t0 = time.time()

            with torch.no_grad():
                output = model(return_loss=False, rescale=True, **input_data)
            
            
            # print('inference time', round((time.time() - t0) * 1000, 2), 'ms')
            infer_time = round((time.time() - t0) * 1000, 2)
            infer_time_arr.append(infer_time)
            # =============== 模型結果送到 GUI 的資料處理 ================

            t0 = time.time()
            data_obj = []
            data_dot = []

            result = output[0]['img_bbox']
            bboxes = result['boxes_3d']

            # 處理 3D bboxes

            for i in range(result['labels_3d'].shape[0]):
                score = result['scores_3d'][i]
                # print(score)
                if score < SCORE_THRESH: 
                    continue
                
                corners = box3d_to_corners(bboxes)[i, [0, 3, 7, 4, 0]]


                # draw front center line
                forward_center = np.mean(corners[2:4], axis=0)
                center = np.mean(corners[0:4], axis=0)
                x = [forward_center[0], center[0]]
                y = [forward_center[1], center[1]]
                
                # x = [(i / 60 + 0.5) * 256 for i in x]
                # y = [(1 - (i / 60 + 0.5)) * 256 for i in y]
                center1 = center
                center = center / 60
                

                # calculate angle degree
                angle = np.arctan2(y[1] - y[0], x[1] - x[0]) * 180 / np.pi

                # car stopped or not
                is_stop = 0
                if IS_TRAJ:
                    traj_score = result['trajs_score'][i].numpy()
                    traj = result['trajs_3d'][i].numpy()
                    num_modes = len(traj_score)
                    center_for_stop = bboxes[i, :2][None, None].repeat(num_modes, 1, 1).numpy()
                    traj = np.concatenate([center_for_stop, traj], axis=1)

                    sorted_ind = np.argsort(traj_score)[::-1]
                    sorted_traj = traj[sorted_ind, :, :2]
                    sorted_score = traj_score[sorted_ind]
                    norm_score = np.exp(sorted_score[0])
                    
                    
                    viz_traj = sorted_traj[0]
                    traj_score = np.exp(sorted_score[0])/norm_score

                    head2end_dis = int(np.linalg.norm(viz_traj[0] - viz_traj[-1]))

                    if head2end_dis < 30:
                        is_stop = 1
                    else:
                        is_stop = 0
                
                data_obj.append({'x': center[0], 
                                 'y' :-center[1], 
                                 'cls' : CLASS[result['labels_3d'][i]], 
                                 'ang' : angle - 90,
                                 'is_stop' : is_stop})
                
                
            # 處理 vector map
           
            if IS_MAP:
                for i in range(result['scores'].shape[0]):
                    score = result['scores'][i]
                    
                    if score < MAP_SCORE_THRESH:
                        continue
                    # print(score)
                

            
                    pts = result['vectors'][i].copy()

                    pts[:, 0] = (1 - (pts[:, 0] / 60 + 0.5))  # x
                    pts[:, 1] = ((pts[:, 1] / 60 + 0.5))  # y

                    # 將道路類別與原本的設定配對一致
                    # 0 : side, 1 : crosswalk, 2 : roadline
                    if result['labels'][i] == 0: 
                        result['labels'][i] = 1
                    elif result['labels'][i] == 1:
                        result['labels'][i] = 2
                    else:
                        result['labels'][i] = 0

                    data_dot.append({'x' : pts[:, 0].tolist(), 
                                    'y' : pts[:, 1].tolist(), 
                                    'cls' : int(result['labels'][i])}) # x, y, class


            # 處理 motion planning
            traj = None
            if IS_TRAJ: 
                traj_all = draw_planning_pred(result)

                traj = np.mean(traj_all, axis=0)

                traj = traj + 0.5
                traj[:, 0] = -traj[:, 0]
                # traj *= 2
                traj = traj.tolist()

            # steering = can_list[idx]['steering']
            # speed = can_list[idx]['vehicle_speed']
            # steering = 0
            # speed = 0
                
            img_dict = {'CAM_FRONT' : base64.b64encode(cv2.imencode('.jpg', cv2.resize(img_ori_arr[0], (470, 264)))[1]).decode('utf-8'),
                        'CAM_BACK' : base64.b64encode(cv2.imencode('.jpg', cv2.resize(img_ori_arr[3], (470, 264)))[1]).decode('utf-8'),
                        }
            
            data_send = {
                    # 'steering' : steering,
                    # 'speed' : speed,
                    'obj' : data_obj,
                    'img' : img_dict,
                    'dot' : data_dot,
                    'traj' : traj}
        
            data_send = json.dumps(data_send).encode('utf-8')


            if IS_SERVER:

                data_send += ('~').encode('utf-8')
                print('send length : ',len(data_send))

                for i in range(0, len(data_send), MAX_CHUNK_SIZE):
                
                    client_socket.sendall(data_send[i:i+MAX_CHUNK_SIZE])

            if not IS_SWEEP:
            
                sample = nusc.get('sample', sample['next'])
            # 每項右邊空格都是一個 tab
            postpro_time = round((time.time() - t0) * 1000, 2)
            # print('postpro time', round((time.time() - t0)  * 1000, 2), 'ms')
            print(  'frame : ', idx,
                    'prepro time : ', prepro_time, 'ms, ',
                    'infer time : ', infer_time, 'ms, ',
                    'postpro time : ', postpro_time, 'ms')
            idx += 1
            
    print('average inference time : ', np.mean(infer_time_arr[1:]), 'ms')

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method(
    #     "fork"
    # )  # use fork workers_per_gpu can be > 1
    main()
