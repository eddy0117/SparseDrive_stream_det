CLASS = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
CLASS_RANGE = {'car': 50, 'truck': 50, 'bus': 50, 'trailer': 50, 'construction_vehicle': 50, 'pedestrian': 40, 'motorcycle': 40, 'bicycle': 40, 'traffic_cone': 30, 'barrier': 30}
CONF_TH = 0.3

X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))  # undecoded
CNS, YNS = 0, 1  # centerness and yawness indices in quality
YAW = 6  # decoded