# Sparsedrive in NGC container with streaming prediction

The repo is cloned from SparseDrive offical [HERE](https://github.com/swc-17/SparseDrive).<br>

## 1. Install Jetson container

---

```
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh
```

### 1.1 Modify run.sh

> Make the jetson-container reuseable
> 
1. Delete `—rm`  at line 331 & 354 
2. Comment if block at line row 231
    
    ```python
    # MODIFIED disable display
    # if [ -n "$DISPLAY" ]; then
    # 	echo "### DISPLAY environmental variable is already set: \"$DISPLAY\""
    # 	# give docker root user X11 permissions
    # 	xhost +si:localuser:root || sudo xhost +si:localuser:root
    	
    # 	# enable SSH X11 forwarding inside container (https://stackoverflow.com/q/48235040)
    	# XAUTH=/tmp/.docker.xauth
    	# xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
    	# chmod 777 $XAUTH
    
    # 	DISPLAY_DEVICE="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH"
    # fi
    ```
    

## 2. Run jetson container

---

Container source: 

[https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags)

Container support matrix: 

[https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)

### 2.1 Pull the container

- version: 24.10-py3-igpu
    
    > To fix the system python version to 3.10 because python 3.12 may cause error in some ways when using pip install.
    > 
    
    ```
    jetson-containers run -v <path/to/host/sparsedrive>:<path/to/container/sparsedrive> nvcr.io/nvidia/pytorch:24.10-py3-igpu
    ```
    

## 3. In jetson-container

---

Don’t forget to change directory to sparsedrive workdir.

### 3.1 Install necessery package

```
apt update
apt install libgeos-dev # (for python shapely module)
```

### 3.2 Install DFA operator

```
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
```

### 3.3 Install flash attention

1. `git clone [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)`
2. modify row 179 

```python
# if "80" in cuda_archs():
cc_flag.append("-gencode")
cc_flag.append("arch=compute_87,code=sm_87")  # modified
```

1. `git config --global --add [safe.directory](http://safe.directory) "*"`
2. `python [setup.py](http://setup.py) install`
    
    > This may take a long time on Jetson AGX Orin.
    > 

### 3.4 Install other dependencies

- mmcv-full==1.7.2
- others in requirements.txt

## Frequently problems

---

### cv2.dnn has no attribute ‘DictValue’

- comment line 169 in  “/usr/local/lib/python3.10/dist-packages/cv2/typing/**init**.py”

```python
#  LayerId = cv2.dnn.DictValue
```

src :  [https://github.com/facebookresearch/nougat/issues/40](https://github.com/facebookresearch/nougat/issues/40)


