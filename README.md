<h1 align="center">AirSLAM: An Efficient and Illumination-Robust Point-Line Visual SLAM System</h1>

<p align="center"><strong>
    <a href = "https://scholar.google.com/citations?user=-p7HvCMAAAAJ&hl=zh-CN">Kuan Xu</a><sup>1</sup>,
    <a href = "https://github.com/yuefanhao">Yuefan Hao</a><sup>2</sup>,
    <a href = "https://scholar.google.com/citations?user=XcV_sesAAAAJ&hl=en">Shenghai Yuan</a><sup>1</sup>,
    <a href = "https://sairlab.org/team/chenw/">Chen Wang</a><sup>2</sup>,
    <a href = "https://scholar.google.com.sg/citations?user=Fmrv3J8AAAAJ&hl=en">Lihua Xie</a><sup>1</sup>
</strong></p>

<p align="center"><strong>
    <a href = "https://www.ntu.edu.sg/cartin">1: Centre for Advanced Robotics Technology Innovation (CARTIN), Nanyang Technological University</a><br>
    <a href = "https://sairlab.org/">2: Spatial AI & Robotics (SAIR) Lab, Computer Science and Engineering, University at Buffalo</a><br>
</strong></p>

<p align="center"><strong> 
    <a href = "https://arxiv.org/pdf/2408.03520">&#128196; [PDF]</a> | 
    <a href = "https://xukuanhit.github.io/airslam/">&#128190; [Project Site]</a> |
    <a href = "https://youtu.be/5OcR5KeO5nc">&#127909; [Youtube]</a> |
    <a href = "https://www.bilibili.com/video/BV1rJY7efE9x">&#127909; [Bilibili]</a>
    <!-- &#128214; [OpenAccess] -->
</strong></p>

### Accepted to IEEE Transactions on Robotics (TRO), 2025

### :scroll: AirSLAM has dual-mode (V-SLAM, VI-SLAM), upgraded from [AirVO (IROS'23)](https://github.com/sair-lab/AirSLAM/releases/tag/1.0)

<p align="middle">
  <img src="figures/system_arch.jpg" width="600" />
</p>

**AirSLAM** is an efficient visual SLAM system designed to tackle both short-term and long-term illumination
challenges. Our system adopts a hybrid approach that combines deep learning techniques for feature detection and matching with traditional backend optimization methods. Specifically, we propose a unified convolutional neural network (CNN) that simultaneously extracts keypoints and structural lines. These features are then associated, matched, triangulated, and optimized in a coupled manner. Additionally, we introduce a lightweight relocalization pipeline that reuses the built map, where keypoints, lines, and a structure graph are used to match the query frame with the map. To enhance the applicability of the proposed system to real-world robots, we deploy and accelerate the feature detection and matching networks using C++ and NVIDIA TensorRT. Extensive experiments conducted on various datasets demonstrate that our system outperforms other state-of-the-art visual SLAM systems in illumination-challenging environments. Efficiency evaluations show that our system can run at a rate of 73Hz on a PC and 40Hz on an embedded platform.

**Video**
<p align="middle">
<a href="https://youtu.be/5OcR5KeO5nc" target="_blank"><img src="figures/title.JPG" width="600" border="10"/></a>
</p>


## :eyes: Updates
* [2025.01] The paper [AirSLAM](https://arxiv.org/pdf/2408.03520) was officially accepted to IEEE Transactions on Robotics (TRO).
* [2025.01] We release the training code for PLNet. The Python code for PLNet can now be found [here](https://github.com/sair-lab/PLNet).
* [2024.08] We release the code and paper for AirSLAM.
* [2023.07] AriVO is accepted by IROS 2023.
* [2022.10] We release the code and paper for AirVO. The code for AirVO can now be found [here](https://github.com/sair-lab/AirSLAM/tree/airvo_iros).


## :checkered_flag: Test Environment
### Dependencies
* OpenCV 4.2
* Eigen 3
* Ceres 2.0.0
* G2O (tag:20230223_git)
* TensorRT 8.6.1.6
* CUDA 12.1
* python
* ROS noetic
* Boost

### Docker (Recommend)
```bash
docker pull xukuanhit/air_slam:v4
docker run -it --env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix --privileged --runtime nvidia --gpus all --volume ${PWD}:/workspace --workdir /workspace --name air_slam xukuanhit/air_slam:v4 /bin/bash
```

## :book: Data
The data for mapping should be organized in the following Autonomous Systems Lab (ASL) dataset format (imu data is optional):

```
dataroot
├── cam0
│   └── data
│       ├── t0.jpg
│       ├── t1.jpg
│       ├── t2.jpg
│       └── ......
├── cam1
│   └── data
│       ├── t0.jpg
│       ├── t1.jpg
│       ├── t2.jpg
│       └── ......
└── imu0
    └── data.csv

```
After the map is built, the relocalization requires only monocular images. Therefore, you only need to place the query images in a folder.


## :computer: Build
```
    cd ~/catkin_ws/src
    git clone https://github.com/sair-lab/AirSLAM.git
    cd ../
    catkin_make
    source ~/catkin_ws/devel/setup.bash
```

## :running: Run 

The launch files for VO/VIO, map optimization, and relocalization are placed in [VO folder](launch/visual_odometry), [MR folder](launch/map_refinement), and [Reloc folder](launch/relocalization), respectively. Before running them, you need to modify the corresponding configurations according to your data path and the desired map-saving path. The following is an example of mapping, optimization, and relocalization with the EuRoC dataset.  


### Mapping
**1**: Change "dataroot" in [VO launch file](launch/visual_odometry/vo_euroc.launch) to your own data path. For the EuRoC dataset, "mav0" needs to be included in the path.

**2**: Change "saving_dir" in the same file to the path where you want to save the map and trajectory. **It must be an existing folder.**

**3**: Run the launch file:

```
roslaunch air_slam vo_euroc.launch 
```

### Map Optimization
**1**: Change "map_root" in [MR launch file](launch/map_refinement/mr_euroc.launch) to your own map path.

**2**: Run the launch file:

```
roslaunch air_slam mr_euroc.launch 
```

### Relocalization
**1**: Change "dataroot" in [Reloc launch file](launch/relocalization/reloc_euroc.launch) to your own query data path.

**2**: Change "map_root" in the same file to your own map path.

**3**: Run the launch file:

```
roslaunch air_slam reloc_euroc.launch 
```

### Other datasets
[Launch folder](launch) and [config folder](configs) respectively provide the launch files and configuration files for other datasets in the paper. If you want to run AirSLAM with your own dataset, you need to create your own camera file, configuration file, and launch file. 


## :writing_hand: TODO List

- [x] Initial release. :rocket:
- [ ] Support more GPUs and development environments
- [ ] Support SuperGlue as the feature matcher
- [ ] Optimize the TensorRT acceleration of PLNet
- [ ] Optimize the evaluation using [PyPose](https://pypose.org/docs/main/metric/)


## :pencil: Citation
```bibtex
@article{xu2024airslam,
  title = {{AirSLAM}: An Efficient and Illumination-Robust Point-Line Visual SLAM System},
  author = {Xu, Kuan and Hao, Yuefan and Yuan, Shenghai and Wang, Chen and Xie, Lihua},
  journal = {IEEE Transactions on Robotics (TRO)},
  year = {2024},
  url = {https://arxiv.org/abs/2408.03520},
  code = {https://github.com/sair-lab/AirSLAM},
}

@inproceedings{xu2023airvo,
  title = {{AirVO}: An Illumination-Robust Point-Line Visual Odometry},
  author = {Xu, Kuan and Hao, Yuefan and Yuan, Shenghai and Wang, Chen and Xie, Lihua},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year = {2023},
  url = {https://arxiv.org/abs/2212.07595},
  code = {https://github.com/sair-lab/AirVO},
  video = {https://youtu.be/YfOCLll_PfU},
}
```

## :checkered_flag: Test Environment
### Dependencies
* OpenCV 4.2
* Eigen 3
* Ceres 2.0.0
* G2O (tag:20230223_git)
* TensorRT 8.6.1.6
* CUDA 12.1
* python
* ROS noetic
* Boost

### Docker (Recommend)
```bash
docker pull xukuanhit/air_slam:v4
docker run -it --env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix --privileged --runtime nvidia --gpus all --volume ${PWD}:/workspace --workdir /workspace --name air_slam xukuanhit/air_slam:v4 /bin/bash
```

## :book: Supported Datasets

### EuRoC MAV Dataset

**Dataset Overview**

The [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) is a comprehensive collection of visual-inertial data recorded onboard a Micro Aerial Vehicle (MAV). Provided by the Autonomous Systems Lab (ASL) at ETH Zurich, this dataset is specifically designed for evaluating visual-inertial odometry and SLAM algorithms.

**Dataset Features:**
- **Sensors:**
  - Stereo cameras (Aptina MT9V034, global shutter, WVGA monochrome, 20 fps)
  - ADIS16448 IMU (200 Hz)
  - High-precision ground truth from Vicon motion capture system or Leica MS50
  
- **Environments:**
  - Machine Hall (MH): Large industrial environments with rich textures
  - Vicon Room (V1, V2): Smaller rooms with varying lighting and rapid motions

- **Sequences:** 11 sequences with varying difficulty levels (easy, medium, difficult)

**Download Instructions:**

1. Visit the [official download page](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
2. Download sequences in ASL Dataset Format (.bag files or extracted folders)
3. Recommended sequences for getting started:
   - **Easy:** `MH_01_easy`, `V1_01_easy`
   - **Medium:** `MH_03_medium`, `V2_02_medium`
   - **Difficult:** `MH_05_difficult`, `V2_03_difficult`

**Quick Download via Command Line:**
```bash
# Example: Download MH_01_easy sequence
wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip
unzip MH_01_easy.zip
```

### Dataset Structure

The data should be organized in the Autonomous Systems Lab (ASL) dataset format. Both custom datasets and standard benchmark datasets (like EuRoC) should follow this structure:

**For Visual-Inertial SLAM (with IMU):**
```
dataroot
├── mav0/                          # EuRoC format (optional prefix)
│   ├── cam0/                      # Left camera
│   │   ├── data/                  # Image data folder
│   │   │   ├── 1403636579763555584.png
│   │   │   ├── 1403636579813555456.png
│   │   │   └── ...
│   │   ├── data.csv              # Image timestamps
│   │   └── sensor.yaml           # Camera calibration
│   ├── cam1/                      # Right camera (for stereo)
│   │   ├── data/
│   │   │   ├── 1403636579763555584.png
│   │   │   ├── 1403636579813555456.png
│   │   │   └── ...
│   │   ├── data.csv
│   │   └── sensor.yaml
│   ├── imu0/                      # IMU data
│   │   ├── data.csv              # IMU measurements
│   │   └── sensor.yaml           # IMU parameters
│   ├── state_groundtruth_estimate0/  # Ground truth (optional)
│   │   └── data.csv
│   └── body.yaml                 # Sensor extrinsics
```

**Simplified Structure (without mav0 prefix):**
```
dataroot
├── cam0
│   └── data
│       ├── t0.png
│       ├── t1.png
│       ├── t2.png
│       └── ......
├── cam1
│   └── data
│       ├── t0.png
│       ├── t1.png
│       ├── t2.png
│       └── ......
└── imu0
    └── data.csv
```

**For Relocalization (monocular only):**
```
query_folder/
├── query_image_0.png
├── query_image_1.png
├── query_image_2.png
└── ...
```

### Data File Formats

**Camera Data (cam0/data.csv, cam1/data.csv):**
```csv
#timestamp [ns],filename
1403636579763555584,1403636579763555584.png
1403636579813555456,1403636579813555456.png
```

**IMU Data (imu0/data.csv):**
```csv
#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
1403636579758555392,0.0127458,-0.0523806,0.00275179,8.17236,-0.38726,-3.64855
1403636579763555392,-0.00924422,-0.0380217,0.00596769,8.30265,-0.18938,-3.96351
```

**Ground Truth Data (state_groundtruth_estimate0/data.csv):**
```csv
#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],v_RS_R_z [m s^-1],b_w_RS_S_x [rad s^-1],b_w_RS_S_y [rad s^-1],b_w_RS_S_z [rad s^-1],b_a_RS_S_x [m s^-2],b_a_RS_S_y [m s^-2],b_a_RS_S_z [m s^-2]
1403636579758555392,4.688,-1.786,0.783,0.534,-0.153,0.768,0.318,0.047,0.010,-0.002,-0.013,-0.053,0.003,0.172,-0.387,-0.165
```

### Dataset Statistics

| Sequence | Duration | Trajectory Length | Difficulty | Key Features |
|----------|----------|-------------------|------------|--------------|
| MH_01_easy | 182s | 80.6m | Easy | Slow motion, good illumination |
| MH_02_easy | 150s | 73.8m | Easy | Slow motion, good illumination |
| MH_03_medium | 132s | 130.9m | Medium | Medium speed |
| MH_04_difficult | 99s | 91.7m | Difficult | Fast motion |
| MH_05_difficult | 111s | 97.6m | Difficult | Fast motion |
| V1_01_easy | 143s | 58.4m | Easy | Slow motion |
| V1_02_medium | 83s | 75.9m | Medium | Medium speed |
| V1_03_difficult | 105s | 79.4m | Difficult | Fast, pure rotation |
| V2_01_easy | 112s | 36.5m | Easy | Slow motion |
| V2_02_medium | 115s | 83.2m | Medium | Medium speed |
| V2_03_difficult | 115s | 86.1m | Difficult | Fast motion |

### Other Supported Datasets

AirSLAM also supports the following datasets with corresponding configuration files in the [configs](configs) folder:
- **TUM-VI Dataset**
- **OIVIO Dataset** (custom illumination-challenging dataset)
- **Custom datasets** (following ASL format)

For custom datasets, you need to:
1. Organize your data following the ASL format
2. Create a camera calibration file (`.yaml`)
3. Create a configuration file in the `configs` folder
4. Create a launch file in the `launch` folder

Refer to the [EuRoC configuration](configs/euroc) as an example.

## :computer: Build
```bash
cd ~/catkin_ws/src
git clone https://github.com/sair-lab/AirSLAM.git
cd ../
catkin_make
source ~/catkin_ws/devel/setup.bash
```

## :running: Run 

The launch files for VO/VIO, map optimization, and relocalization are placed in [VO folder](launch/visual_odometry), [MR folder](launch/map_refinement), and [Reloc folder](launch/relocalization), respectively. Before running them, you need to modify the corresponding configurations according to your data path and the desired map-saving path. The following is an example of mapping, optimization, and relocalization with the EuRoC dataset.  

### Running with EuRoC Dataset

#### Step 1: Prepare Dataset
```bash
# Download and extract EuRoC dataset
wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip
unzip MH_01_easy.zip -d /path/to/datasets/

# Your dataset structure should be:
# /path/to/datasets/MH_01_easy/mav0/
#   ├── cam0/
#   ├── cam1/
#   └── imu0/
```

#### Step 2: Mapping

**1**: Change "dataroot" in [VO launch file](launch/visual_odometry/vo_euroc.launch) to your own data path. For the EuRoC dataset, "mav0" needs to be included in the path.
```xml
<!-- Example path configuration -->
<param name="dataroot" type="string" value="/path/to/datasets/MH_01_easy/mav0" />
```

**2**: Change "saving_dir" in the same file to the path where you want to save the map and trajectory. **It must be an existing folder.**
```bash
# Create output directory
mkdir -p /path/to/output/MH_01_easy
```

**3**: Run the launch file:
```bash
roslaunch air_slam vo_euroc.launch 
```

The system will process the sequence and save:
- Trajectory file (TUM format)
- Map database
- Feature visualization (if enabled)

#### Step 3: Map Optimization

**1**: Change "map_root" in [MR launch file](launch/map_refinement/mr_euroc.launch) to your own map path.
```xml
<param name="map_root" type="string" value="/path/to/output/MH_01_easy" />
```

**2**: Run the launch file:
```bash
roslaunch air_slam mr_euroc.launch 
```

This will perform global bundle adjustment and loop closure optimization on the built map.

#### Step 4: Relocalization

**1**: Change "dataroot" in [Reloc launch file](launch/relocalization/reloc_euroc.launch) to your own query data path.
```xml
<!-- Can be a different sequence or subset of frames -->
<param name="dataroot" type="string" value="/path/to/query/images" />
```

**2**: Change "map_root" in the same file to your own map path.
```xml
<param name="map_root" type="string" value="/path/to/output/MH_01_easy" />
```

**3**: Run the launch file:
```bash
roslaunch air_slam reloc_euroc.launch 
```

### Running with Other Datasets

[Launch folder](launch) and [config folder](configs) respectively provide the launch files and configuration files for other datasets in the paper. If you want to run AirSLAM with your own dataset, you need to:

1. **Create a camera calibration file** (`camera.yaml`):
```yaml
# Camera intrinsics and distortion
intrinsics: [fx, fy, cx, cy]
distortion_coefficients: [k1, k2, p1, p2]
resolution: [width, height]
camera_model: pinhole  # or omni

# Stereo baseline (if using stereo)
T_cn_cnm1:
  - [R11, R12, R13, tx]
  - [R21, R22, R23, ty]
  - [R31, R32, R33, tz]
  - [0.0, 0.0, 0.0, 1.0]
```

2. **Create a configuration file** in `configs/your_dataset/`:
```yaml
# System configuration
use_imu: true
use_line: true
max_keypoints: 300
# ... other parameters
```

3. **Create launch files** in `launch/` folders following the EuRoC examples

4. **Organize your data** following the ASL format structure shown above

### Visualization and Evaluation

During execution, you can visualize:
- Feature detection and matching in RViz
- Real-time trajectory
- Map points and lines

After completion, evaluate results using:
```bash
# Install evo for trajectory evaluation
pip install evo

# Evaluate against ground truth
evo_ape tum groundtruth.txt trajectory.txt -va --plot --plot_mode xy
evo_rpe tum groundtruth.txt trajectory.txt -va --plot
```

## :writing_hand: TODO List

- [x] Initial release. :rocket:
- [x] Support for EuRoC MAV Dataset
- [ ] Support more GPUs and development environments
- [ ] Support SuperGlue as the feature matcher
- [ ] Optimize the TensorRT acceleration of PLNet
- [ ] Optimize the evaluation using [PyPose](https://pypose.org/docs/main/metric/)
- [ ] ROS 2 support
- [ ] Real-time dataset player


## :pencil: Citation
```bibtex
@article{xu2024airslam,
  title = {{AirSLAM}: An Efficient and Illumination-Robust Point-Line Visual SLAM System},
  author = {Xu, Kuan and Hao, Yuefan and Yuan, Shenghai and Wang, Chen and Xie, Lihua},
  journal = {IEEE Transactions on Robotics (TRO)},
  year = {2024},
  url = {https://arxiv.org/abs/2408.03520},
  code = {https://github.com/sair-lab/AirSLAM},
}

@inproceedings{xu2023airvo,
  title = {{AirVO}: An Illumination-Robust Point-Line Visual Odometry},
  author = {Xu, Kuan and Hao, Yuefan and Yuan, Shenghai and Wang, Chen and Xie, Lihua},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year = {2023},
  url = {https://arxiv.org/abs/2212.07595},
  code = {https://github.com/sair-lab/AirVO},
  video = {https://youtu.be/YfOCLll_PfU},
}
```

## :link: Useful Links

- **EuRoC Dataset:** https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
- **TUM-VI Dataset:** https://vision.in.tum.de/data/datasets/visual-inertial-dataset
- **EVO Evaluation Tool:** https://github.com/MichaelGrupp/evo
- **Kalibr Calibration:** https://github.com/ethz-asl/kalibr

## :page_facing_up: License

This project is released under the [MIT License](LICENSE).

## :handshake: Acknowledgments

We thank ETH Zurich ASL for providing the EuRoC MAV Dataset and the open-source community for valuable tools and libraries.
