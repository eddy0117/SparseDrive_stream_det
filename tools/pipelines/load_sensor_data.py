from pyquaternion import Quaternion
from .utils.pre_process import *

def load_sensor_data(nusc, sample, nusc_can, idx, IS_SWEEP=False):

    if idx == 0 or not IS_SWEEP:

        sd_rec_l = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

    elif IS_SWEEP:

        sd_rec_l = nusc.get('sample_data', sd_rec_l['next'])

    cs_record_l = nusc.get('calibrated_sensor', sd_rec_l['calibrated_sensor_token'])

    pose_record_l = nusc.get('ego_pose', sd_rec_l['ego_pose_token'])

    timestamp = sd_rec_l['timestamp'] / 1e6

    e2g_rotation = Quaternion(pose_record_l['rotation']).rotation_matrix
    e2g_translation = pose_record_l['translation']
    l2e_rotation = Quaternion(cs_record_l['rotation']).rotation_matrix
    l2e_translation = cs_record_l['translation']
    e2g_matrix = convert_egopose_to_matrix_numpy(e2g_rotation, e2g_translation)
    l2e_matrix = convert_egopose_to_matrix_numpy(l2e_rotation, l2e_translation)

    ego_pose = e2g_matrix @ l2e_matrix
    ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)

    # ego_status

    ego_status = get_ego_status(nusc, nusc_can, sample)

    return {
        'ego_pose': ego_pose,
        'ego_pose_inv': ego_pose_inv,
        'timestamp': timestamp,
        'ego_status': ego_status,
        'sd_rec_l': sd_rec_l,
        'cs_record_l': cs_record_l,
        'pose_record_l': pose_record_l,
        'e2g_rotation': e2g_rotation,
        'e2g_translation': e2g_translation,
        'l2e_rotation': l2e_rotation,
        'l2e_translation': l2e_translation
    }