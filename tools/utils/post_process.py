
import pyquaternion 
import numpy as np
import torch
import cv2
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box as NuScenesBox
from .global_var import *

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


# motion planning

def draw_planning_pred(result, top_k=3):
    
    # import ipdb; ipdb.set_trace()
    plan_trajs = result['planning'].cpu().numpy()
    num_cmd = 3
    num_mode = plan_trajs.shape[1]
    plan_trajs_ = np.concatenate((np.zeros((num_cmd, num_mode, 1, 2)), plan_trajs), axis=2)
    plan_score_ = result['planning_score'].cpu().numpy()

    # cmd = data['gt_ego_fut_cmd'].argmax()
    result = []
    for cmd in range(3):
        plan_trajs = plan_trajs_[cmd]
        plan_score = plan_score_[cmd]

        sorted_ind = np.argsort(plan_score)[::-1]
      
        sorted_traj = plan_trajs[sorted_ind, :, :2]
        sorted_score = plan_score[sorted_ind]
        norm_score = np.exp(sorted_score[0])
        
        for j in range(top_k - 1, -1, -1):
            viz_traj = sorted_traj[j]
            traj_score = np.exp(sorted_score[j]) / norm_score
        result.append(render_traj(viz_traj, traj_score=traj_score,
                        colormap='autumn', dot_size=50))
    return result

def render_traj(
    future_traj, 
    traj_score=1, 
    colormap='winter', 
    points_per_step=20, 
    dot_size=25
):
    total_steps = (len(future_traj) - 1) * points_per_step + 1
    # dot_colors = matplotlib.colormaps[colormap](
    #     np.linspace(0, 1, total_steps))[:, :3]
    # dot_colors = dot_colors * traj_score + \
    #     (1 - traj_score) * np.ones_like(dot_colors)
    total_xy = np.zeros((total_steps, 2))
    for i in range(total_steps - 1):
        unit_vec = future_traj[i // points_per_step +
                                1] - future_traj[i // points_per_step]
        total_xy[i] = (i / points_per_step - i // points_per_step) * \
            unit_vec + future_traj[i // points_per_step]
    total_xy[-1] = future_traj[-1]

    return total_xy