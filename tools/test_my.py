# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import mmcv
import os
import torch
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import (load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.models import build_model
from mmdet.apis import set_random_seed

from mmdet.datasets import replace_ImageToTensor
from nuscenes.utils.geometry_utils import view_points
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from mmdetection3d.mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes


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

VERSION = 'v1.0-mini'

MAP_SIZE = 682

IS_SWEEP = False

IS_SERVER = False

set_random_seed(42)

det_grid_conf = {
    'xbound': [-51.2, 51.2, 0.68],
    'ybound': [-51.2, 51.2, 0.68],
}

map_grid_conf = {
    'xbound': [-30.0, 30.0, 0.15],
    'ybound': [-15.0, 15.0, 0.15],
}



def main():
    

    lidar2img_fixed_dict = {}
    config_path = 'projects/configs/StreamPETR/stream_petr_vov_flash_800_bs8_seq_24e_mod.py'
    # checkpoint_path = 'work_dirs/stream_petr_vov_flash_800_bs2_seq_24e/latest.pth'
    checkpoint_path = 'ckpts/seg.pth'
    samples_per_gpu = 1

    # socket connection setup

    if IS_SERVER:

        print('connecting to server')
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 65432))
        print('finish connecting')
    
    # checkpoint_path = 'ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth'
    
    mean = np.array([103.530, 116.280, 123.675], dtype=np.float32)
    std = np.array([57.375, 57.120, 58.395], dtype=np.float32)

    scene_list = ['scene-0103', 'scene-0916']
    # scene_list = ['scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016']
    # scene_list = ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100']
    
    cams = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    bev_files = sorted(os.listdir('tools/gt_t'), key=lambda x: int(x.split('.')[0]))
    
    sd_rec_c_dict = {}

    

    bev_idx = 0
    
    cfg = Config.fromfile(config_path)
    
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    
    # model = fuse_conv_bn(model)

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
   
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
   

  
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

        img_metas = {'pad_shape' : [(320, 800, 3), (320, 800, 3), (320, 800, 3), (320, 800, 3), (320, 800, 3), (320, 800, 3)],
                    'box_type_3d' : LiDARInstance3DBoxes,
                    'scene_token' : scene['token']}

        sample = nusc.get('sample', scene['first_sample_token'])

        

        while sample['next'] != '':

            t0 = time.time()

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

            # ego pose

            # cs_record['translation'] -> lidar2ego_translation
            # cs_record['rotation'] -> lidar2ego_rotation
            # pose_record['translation'] -> ego2global_translation
            # pose_record['rotation'] -> ego2global_rotation

            # add for bboxes convert to global
            # box_convert_dict = {'lidar2ego_translation' : cs_record_l['translation'],
            #                     'lidar2ego_rotation' : cs_record_l['rotation'],
            #                     'ego2global_translation' : pose_record_l['translation'],
            #                     'ego2global_rotation' : pose_record_l['rotation']}

            e2g_rotation = Quaternion(pose_record_l['rotation']).rotation_matrix
            e2g_translation = pose_record_l['translation']
            l2e_rotation = Quaternion(cs_record_l['rotation']).rotation_matrix
            l2e_translation = cs_record_l['translation']
            e2g_matrix = convert_egopose_to_matrix_numpy(e2g_rotation, e2g_translation)
            l2e_matrix = convert_egopose_to_matrix_numpy(l2e_rotation, l2e_translation)

            ego_pose = e2g_matrix @ l2e_matrix
            ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
            
            

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
                l2e_t = np.array(cs_record_l['translation'])
                e2g_t = np.array(pose_record_l['translation'])
                l2e_r = np.array(cs_record_l['rotation'])
                e2g_r = np.array(pose_record_l['rotation'])

                # cam
                l2e_t_s = np.array(cs_record_c['translation'])
                e2g_t_s = np.array(pose_record_c['translation'])
                l2e_r_s = np.array(cs_record_c['rotation'])
                e2g_r_s = np.array(pose_record_c['rotation'])


                l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                e2g_r_mat = Quaternion(e2g_r).rotation_matrix

                l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
                e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

                R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
                        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
                    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                                ) + l2e_t @ np.linalg.inv(l2e_r_mat).T

                cam2lidar_r = R.T
                cam2lidar_t = T


                cam2lidar_rt = convert_egopose_to_matrix_numpy(cam2lidar_r, cam2lidar_t)
                lidar2cam_rt = invert_matrix_egopose_numpy(cam2lidar_rt)

                intrinsic = np.array(cs_record_c['camera_intrinsic'])

                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

                lidar2img_rt = (viewpad @ lidar2cam_rt)

                intrinsic = viewpad
                extrinsic = lidar2cam_rt
                
                
                # LoadImage
                
                img_name = 'data/nuscenes/' + sd_rec_c['filename']
                img_ori = mmcv.imread(img_name, 'unchanged')
                img_ori_arr.append(img_ori)
                
                # ResizeCropFlipRotImage class call
                
                # t0 = time.time()
                img, ida_mat = img_transform(
                                img_ori,
                                resize=0.5,
                                resize_dims=(800, 450),
                                crop=(0, 130, 800, 450),
                            )
                intrinsic[:3, :3] = ida_mat @ intrinsic[:3, :3]
                
                lidar2img = intrinsic @ extrinsic
                
                #===================================
                # PadImage

                img = mmcv.impad_to_multiple(img, 32, pad_val=0)
                
                # NormalizeImage
                
                img = mmcv.imnormalize(img, mean, std, False)

                intrinsic_arr.append(intrinsic)
                extrinsic_arr.append(extrinsic)
                lidar2img_arr.append(lidar2img)
                filename_arr.append('data/nuscenes/' + sd_rec_c['filename'])
                img_arr.append(img.transpose(2, 0, 1))
                # print('preprocess time', time.time() - t0)

            img_arr = np.stack(img_arr)

        

            input_data = {'img_metas': [[img_metas]],
                        'img': [torch.Tensor(img_arr).unsqueeze(0).to('cuda')],
                        'intrinsics' : [[torch.Tensor(intrinsic_arr).to('cuda')]],
                        'extrinsics' : [[torch.Tensor(extrinsic_arr).to('cuda')]],
                        'lidar2img' : [[torch.Tensor(lidar2img_arr).to('cuda')]],
                        'ego_pose' : [[torch.Tensor(ego_pose).to('cuda')]],
                        'ego_pose_inv' : [[torch.Tensor(ego_pose_inv).to('cuda')]],
                        'timestamp' : [torch.Tensor([timestamp]).to('cuda')]}

        
            

            

            # t0 = time.time()
            with torch.no_grad():
                output = model(return_loss=False, rescale=True, **input_data)
                
            
            result_list = output_to_nusc_box(output[0]['pts_bbox'])
            # result_list = lidar_nusc_box_to_global(box_convert_dict, result_list)
            


            view = np.array([[MAP_SIZE // (det_grid_conf['xbound'][1] * 2), 0, 0, MAP_SIZE / 2],
                                    [0, -MAP_SIZE // (det_grid_conf['xbound'][1] * 2), 0, MAP_SIZE / 2],
                                    [0, 0, 1, 0], [0, 0, 0, 1]])
            
            # prepare objects data

            data_obj = []

            for box in result_list:
                corners = view_points(box.corners(), view, normalize=False)[:2, :]
                center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
                data_obj.append({'x': int(center_bottom[0]), 
                                 'y' : int(center_bottom[1]), 
                                 'cls' : CLASS[box.label], 
                                 'ang' : box.orientation.degrees + 90})
                
            bev_img = mmcv.imread('tools/gt_t/' + bev_files[0], 'unchanged')
            steering = can_list[idx]['steering']
            speed = can_list[idx]['vehicle_speed']
            
            img_dict = {'CAM_FRONT' : base64.b64encode(cv2.imencode('.jpg', cv2.resize(img_ori_arr[0], (470, 264)))[1]).decode('utf-8'),
                        'CAM_BACK' : base64.b64encode(cv2.imencode('.jpg', cv2.resize(img_ori_arr[3], (470, 264)))[1]).decode('utf-8'),
                        'BEV' : base64.b64encode(cv2.imencode('.jpg', bev_img)[1]).decode('utf-8')}
            
            data_send = {'steering' : steering,
                        'speed' : speed,
                        'obj' : data_obj,
                        'img' : img_dict}
            
            data_send = json.dumps(data_send).encode('utf-8')
                    
            
            print('prediction time', time.time() - t0)

            if IS_SERVER:

                data_send += ('\0').encode('utf-8')
                print('send length : ',len(data_send))
                for i in range(0, len(data_send), MAX_CHUNK_SIZE):
                
                    client_socket.sendall(data_send[i:i+MAX_CHUNK_SIZE])

            if not IS_SWEEP:
            
                sample = nusc.get('sample', sample['next'])

            idx += 1
            bev_idx += 1

    if IS_SERVER:
        client_socket.close()
    
if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('fork')
    main()
