import numpy as np
from pyquaternion import Quaternion
from .utils.pre_process import convert_egopose_to_matrix_numpy, invert_matrix_egopose_numpy

def process_sensor_data(result_dict: dict, sd_rec_c_dict: dict, nusc, sample, idx, cam, IS_SWEEP):

    if idx == 0 or not IS_SWEEP:
                    
        cam_token = sample['data'][cam]
    

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
    c2e_t_s = np.array(cs_record_c['translation'])
    c2e_r_s = np.array(cs_record_c['rotation'])

    # cam to global
    e2g_t_s = np.array(pose_record_c['translation']) 
    e2g_r_s = np.array(pose_record_c['rotation'])


    # l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    # e2g_r_mat = Quaternion(e2g_r).rotation_matrix
    l2e_r_mat = result_dict['l2e_rotation']
    e2g_r_mat = result_dict['e2g_rotation']
    l2e_t = result_dict['l2e_translation']
    e2g_t = result_dict['e2g_translation']

    c2e_r_s_mat = Quaternion(c2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

    R = (c2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (c2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
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

    # lidar2img
    lidar2img = (viewpad @ lidar2cam_rt)

    intrinsic = viewpad
    extrinsic = lidar2cam_rt

    result_dict.update({
        'sd_rec_c': sd_rec_c,
        'cam2lidar_r': cam2lidar_r,
        'cam2lidar_t': cam2lidar_t,
        'cam2lidar_rt': cam2lidar_rt,
        'lidar2cam_rt': lidar2cam_rt,
        'intrinsic': intrinsic,
        'extrinsic': extrinsic,
        'viewpad': viewpad,
        'lidar2img': lidar2img
    })

    return result_dict