
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.utils.geometry_utils import view_points
import pyquaternion 
import cv2
import numpy as np
import torch

CLASS = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
CLASS_RANGE = {'car': 50, 'truck': 50, 'bus': 50, 'trailer': 50, 'construction_vehicle': 50, 'pedestrian': 40, 'motorcycle': 40, 'bicycle': 40, 'traffic_cone': 30, 'barrier': 30}
CONF_TH = 0.3

X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))  # undecoded
CNS, YNS = 0, 1  # centerness and yawness indices in quality
YAW = 6  # decoded

def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    bottom_center = box3d[:, :3]
    gravity_center = torch.zeros_like(bottom_center)
    gravity_center[:, :2] = bottom_center[:, :2]
    gravity_center[:, 2] = bottom_center[:, 2] + box3d[:, 5] * 0.5

    # box_dims = box3d.dims.numpy()
    box_dims = box3d[:, 3:6]
    # box_yaw = box3d.yaw.numpy()
    box_yaw = box3d[:, 6]

    # our LiDAR coordinate system -> nuScenes box coordinate system
    nus_box_dims = box_dims[:, [1, 0, 2]]

    box_list = []
    for i in range(len(box3d)):

        # filter det in ego.
        if scores[i] < CONF_TH:
            continue


        radius = np.linalg.norm(np.array(gravity_center[i])[:2], 2)
        det_range = CLASS_RANGE[CLASS[labels[i]]]
        if radius > det_range:
            continue

        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])

        velocity = (0, 0, 0)
  
        box = NuScenesBox(
            gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        
        # box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        # box.translate(np.array(info['lidar2ego_translation']))

        
        
          
        # Move box to global coord system
        # box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        # box.translate(np.array(info['ego2global_translation']))


        box_list.append(box)
    return box_list

def lidar_nusc_box_to_global(info,
                             boxes):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        # box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        # box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = CLASS_RANGE
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[CLASS[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        # box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        # box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list


def render( box,
            img,
            view: np.ndarray = np.eye(3),
            normalize: bool = False,
            linewidth: float = 2) -> None:
    """
    Renders the box in the provided Matplotlib axis.
    :param axis: Axis onto which the box should be drawn.
    :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
    :param normalize: Whether to normalize the remaining coordinate.
    :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
        back and sides.
    :param linewidth: Width in pixel of the box sides.
    """
   
    corners = view_points(box.corners(), view, normalize=normalize)[:2, :]
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            cv2.line(img,
                        (int(prev[0]), int(prev[1])),
                        (int(corner[0]), int(corner[1])),
                        color, linewidth)
            prev = corner
    center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    # cv2.line(img,
    #         (int(center_bottom[0]), int(center_bottom[1])),
    #         (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
    #         (255, 0, 0), linewidth)

    # # Draw the sides
    # for i in range(4):
    #     axis.plot([corners.T[i][0], corners.T[i + 4][0]],
    #                 [corners.T[i][1], corners.T[i + 4][1]],
    #                 color=colors[2], linewidth=linewidth)
    for i in range(4):
        cv2.line(img,
                (int(corners.T[i][0]), int(corners.T[i][1])),
                (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                (255, 0, 0), linewidth)
    # # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(corners.T[:4], (255, 0, 0))
    draw_rect(corners.T[4:], (255, 0, 0))
    1
    # # Draw line indicating the front
    # center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    # center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    # axis.plot([center_bottom[0], center_bottom_forward[0]],
    #             [center_bottom[1], center_bottom_forward[1]],
    #             color=colors[0], linewidth=linewidth)

def convert_egopose_to_matrix_numpy(rotation, translation):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix

def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix

# StreamPETR
# def img_transform(img, resize, resize_dims, crop):
#     ida_rot = torch.eye(2)
#     ida_tran = torch.zeros(2)
#     # adjust image
    
#     img = cv2.resize(img, resize_dims)
#     img = img[crop[1]:crop[3], crop[0]:crop[2]]
    
#     # post-homography transformation
#     ida_rot *= resize
#     ida_tran -= torch.Tensor(crop[:2])
    
#     ida_mat = torch.eye(3)
#     ida_mat[:2, :2] = ida_rot
#     ida_mat[:2, 2] = ida_tran
#     return img, ida_mat

# SparseDrive

def img_transform(img, aug_configs):
        H, W = img.shape[:2]
        resize = aug_configs.get("resize", 1)
        resize_dims = (int(W * resize), int(H * resize))
        crop = aug_configs.get("crop", [0, 0, *resize_dims])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)

        origin_dtype = img.dtype
        if origin_dtype != np.uint8:
            min_value = img.min()
            max_vaule = img.max()
            scale = 255 / (max_vaule - min_value)
            img = (img - min_value) * scale
            img = np.uint8(img)

        img = cv2.resize(img, resize_dims)
        img = img[crop[1]:crop[3], crop[0]:crop[2]]
       
        img = np.array(img).astype(np.float32)
        if origin_dtype != np.uint8:
            img = img.astype(np.float32)
            img = img / scale + min_value

        transform_matrix = np.eye(3)
        transform_matrix[:2, :2] *= resize
        transform_matrix[:2, 2] -= np.array(crop[:2])
        if flip:
            flip_matrix = np.array(
                [[-1, 0, crop[2] - crop[0]], [0, 1, 0], [0, 0, 1]]
            )
            transform_matrix = flip_matrix @ transform_matrix
        rotate = rotate / 180 * np.pi
        rot_matrix = np.array(
            [
                [np.cos(rotate), np.sin(rotate), 0],
                [-np.sin(rotate), np.cos(rotate), 0],
                [0, 0, 1],
            ]
        )
        rot_center = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        rot_matrix[:2, 2] = -rot_matrix[:2, :2] @ rot_center + rot_center
        transform_matrix = rot_matrix @ transform_matrix
        extend_matrix = np.eye(4)
        extend_matrix[:3, :3] = transform_matrix
        return img, extend_matrix

def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i

def get_ego_status(nusc, nusc_can_bus, sample):
    ego_status = []
    ref_scene = nusc.get("scene", sample['scene_token'])
    try:
        pose_msgs = nusc_can_bus.get_messages(ref_scene['name'],'pose')
        steer_msgs = nusc_can_bus.get_messages(ref_scene['name'], 'steeranglefeedback')
        pose_uts = [msg['utime'] for msg in pose_msgs]
        steer_uts = [msg['utime'] for msg in steer_msgs]
        ref_utime = sample['timestamp']
        pose_index = locate_message(pose_uts, ref_utime)
        pose_data = pose_msgs[pose_index]
        steer_index = locate_message(steer_uts, ref_utime)
        steer_data = steer_msgs[steer_index]
        ego_status.extend(pose_data["accel"]) # acceleration in ego vehicle frame, m/s/s
        ego_status.extend(pose_data["rotation_rate"]) # angular velocity in ego vehicle frame, rad/s
        ego_status.extend(pose_data["vel"]) # velocity in ego vehicle frame, m/s
        ego_status.append(steer_data["value"]) # steering angle, positive: left turn, negative: right turn
    except:
        ego_status = [0] * 10
    
    return np.array(ego_status).astype(np.float32)



def box3d_to_corners(box3d):
    if isinstance(box3d, torch.Tensor):
        box3d = box3d.detach().cpu().numpy()
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # use relative origin [0.5, 0.5, 0]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box3d[:, None, [W, L, H]] * corners_norm.reshape([1, 8, 3])

    # rotate around z axis
    rot_cos = np.cos(box3d[:, YAW])
    rot_sin = np.sin(box3d[:, YAW])
    rot_mat = np.tile(np.eye(3)[None], (box3d.shape[0], 1, 1))
    rot_mat[:, 0, 0] = rot_cos
    rot_mat[:, 0, 1] = -rot_sin
    rot_mat[:, 1, 0] = rot_sin
    rot_mat[:, 1, 1] = rot_cos
    corners = (rot_mat[:, None] @ corners[..., None]).squeeze(axis=-1)
    corners += box3d[:, None, :3]
    return corners