# EuRoC MAV Dataset 使用指南

## 目录
- [数据集简介](#数据集简介)
- [数据集结构](#数据集结构)
- [Dataset类介绍](#dataset类介绍)
- [使用示例](#使用示例)
- [配置选项](#配置选项)
- [常见问题](#常见问题)

---

## 数据集简介

EuRoC MAV Dataset（European Robotics Challenge Micro Aerial Vehicle Dataset）是由苏黎世联邦理工学院（ETH Zurich）提供的用于视觉惯性里程计（VIO）和SLAM研究的数据集。

### 数据集特点

- **传感器配置**：
  - 双目相机（Aptina MT9V034全局快门，WVGA单色，20fps）
  - ADIS16448 IMU（200Hz）
  - 高精度Vicon或Leica MS50动捕系统提供地面真值
  
- **场景**：室内工业环境和机器间，包含多种运动模式和光照条件

- **数据序列**：
  - Machine Hall (MH01-05)：大场景，纹理丰富
  - Vicon Room (V101-03, V201-03)：小场景，快速运动

### 下载地址

官方网站：https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

---

## 数据集结构
```
dataset_root/
├── mav0/
│   ├── cam0/                      # 左相机
│   │   ├── data/                  # 图像数据
│   │   │   ├── 1403636579763555584.png
│   │   │   ├── 1403636579813555456.png
│   │   │   └── ...
│   │   ├── data.csv              # 图像时间戳
│   │   └── sensor.yaml           # 相机标定参数
│   ├── cam1/                      # 右相机
│   │   ├── data/
│   │   ├── data.csv
│   │   └── sensor.yaml
│   ├── imu0/                      # IMU数据
│   │   ├── data.csv              # IMU测量值
│   │   └── sensor.yaml           # IMU参数
│   ├── leica0/                    # Leica地面真值（部分序列）
│   │   └── data.csv
│   ├── state_groundtruth_estimate0/  # 状态估计地面真值
│   │   └── data.csv
│   └── body.yaml                 # 传感器外参
```

### 数据文件格式

#### cam0/data.csv
```csv
#timestamp [ns],filename
1403636579763555584,1403636579763555584.png
1403636579813555456,1403636579813555456.png
```

#### imu0/data.csv
```csv
#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
1403636579758555392,0.0127458,-0.0523806,0.00275179,8.17236,-0.38726,-3.64855
```

#### state_groundtruth_estimate0/data.csv
```csv
#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],v_RS_R_z [m s^-1],b_w_RS_S_x [rad s^-1],b_w_RS_S_y [rad s^-1],b_w_RS_S_z [rad s^-1],b_a_RS_S_x [m s^-2],b_a_RS_S_y [m s^-2],b_a_RS_S_z [m s^-2]
1403636579758555392,4.688,-1.786,0.783,0.534,-0.153,0.768,0.318,0.047,0.010,-0.002,-0.013,-0.053,0.003,0.172,-0.387,-0.165
```

---

## Dataset类介绍

### 类结构
```cpp
class Dataset {
public:
    // 构造函数
    Dataset(const EuRoCConfig& config);
    Dataset(const std::string& dataroot, bool use_imu = true);
    
    // 数据获取
    bool GetData(size_t idx, cv::Mat& left_image, cv::Mat& right_image, 
                 ImuDataList& batch_imu_data, double& timestamp);
    bool GetMonoData(size_t idx, cv::Mat& image, 
                     ImuDataList& batch_imu_data, double& timestamp);
    
    // 数据集信息
    size_t GetDatasetLength() const;
    void PrintDatasetInfo() const;
    bool ValidateDataset() const;
};
```

### 配置结构
```cpp
struct EuRoCConfig {
    std::string dataroot;              // 数据集根目录
    bool use_imu = true;               // 是否使用IMU
    bool use_stereo = true;            // 是否使用双目
    bool use_ground_truth = false;     // 是否加载地面真值
    
    // 可自定义传感器文件夹名称
    std::string cam0_folder = "mav0/cam0";
    std::string cam1_folder = "mav0/cam1";
    std::string imu_folder = "mav0/imu0";
    std::string ground_truth_folder = "mav0/state_groundtruth_estimate0";
};
```

### 数据结构

#### ImuData
```cpp
struct ImuData {
    double timestamp;           // 时间戳（秒）
    Eigen::Vector3d gyr;       // 角速度 [rad/s]
    Eigen::Vector3d acc;       // 加速度 [m/s^2]
};
```

#### GroundTruthData
```cpp
struct GroundTruthData {
    double timestamp;
    Eigen::Vector3d position;         // 位置
    Eigen::Quaterniond orientation;   // 姿态（四元数）
    Eigen::Vector3d velocity;         // 速度
    Eigen::Vector3d bias_gyr;         // 陀螺仪偏置
    Eigen::Vector3d bias_acc;         // 加速度计偏置
    
    Eigen::Matrix4d GetPose() const;  // 获取4x4位姿矩阵
};
```

---

## 使用示例

### 示例1：基础使用（双目 + IMU）
```cpp
#include "dataset.h"

int main() {
    // 方式1：使用简单构造函数
    std::string dataroot = "/path/to/EuRoC/MH_01_easy";
    Dataset dataset(dataroot, true);  // true表示使用IMU
    
    // 遍历数据集
    size_t dataset_length = dataset.GetDatasetLength();
    std::cout << "Dataset length: " << dataset_length << std::endl;
    
    for (size_t i = 0; i < dataset_length; ++i) {
        cv::Mat left_image, right_image;
        ImuDataList imu_data;
        double timestamp;
        
        if (!dataset.GetData(i, left_image, right_image, imu_data, timestamp)) {
            std::cerr << "Failed to get data at index " << i << std::endl;
            continue;
        }
        
        std::cout << "Frame " << i << ", timestamp: " << std::fixed 
                  << std::setprecision(6) << timestamp << std::endl;
        std::cout << "  Left image size: " << left_image.size() << std::endl;
        std::cout << "  Right image size: " << right_image.size() << std::endl;
        std::cout << "  IMU measurements: " << imu_data.size() << std::endl;
        
        // 处理图像和IMU数据
        // ...
    }
    
    return 0;
}
```

### 示例2：使用配置结构
```cpp
#include "dataset.h"

int main() {
    // 创建配置
    EuRoCConfig config;
    config.dataroot = "/path/to/EuRoC/V1_01_easy";
    config.use_imu = true;
    config.use_stereo = true;
    config.use_ground_truth = true;  // 加载地面真值
    
    // 使用配置创建数据集
    Dataset dataset(config);
    
    // 验证数据集
    if (!dataset.ValidateDataset()) {
        std::cerr << "Dataset validation failed!" << std::endl;
        return -1;
    }
    
    // 打印数据集信息
    dataset.PrintDatasetInfo();
    
    return 0;
}
```

### 示例3：单目模式
```cpp
#include "dataset.h"

int main() {
    EuRoCConfig config;
    config.dataroot = "/path/to/EuRoC/MH_02_easy";
    config.use_imu = true;
    config.use_stereo = false;  // 单目模式
    
    Dataset dataset(config);
    
    for (size_t i = 0; i < dataset.GetDatasetLength(); ++i) {
        cv::Mat image;
        ImuDataList imu_data;
        double timestamp;
        
        if (dataset.GetMonoData(i, image, imu_data, timestamp)) {
            // 处理单目图像
            cv::imshow("Image", image);
            cv::waitKey(1);
        }
    }
    
    return 0;
}
```

### 示例4：使用地面真值
```cpp
#include "dataset.h"

int main() {
    EuRoCConfig config;
    config.dataroot = "/path/to/EuRoC/V2_01_easy";
    config.use_imu = true;
    config.use_stereo = true;
    config.use_ground_truth = true;
    
    Dataset dataset(config);
    
    if (!dataset.HasGroundTruth()) {
        std::cerr << "No ground truth available!" << std::endl;
        return -1;
    }
    
    for (size_t i = 0; i < dataset.GetDatasetLength(); ++i) {
        cv::Mat left_image, right_image;
        ImuDataList imu_data;
        double timestamp;
        GroundTruthData ground_truth;
        
        if (dataset.GetData(i, left_image, right_image, imu_data, 
                           timestamp, ground_truth)) {
            // 获取地面真值位姿
            Eigen::Matrix4d gt_pose = ground_truth.GetPose();
            Eigen::Vector3d position = ground_truth.position;
            Eigen::Quaterniond orientation = ground_truth.orientation;
            
            std::cout << "Frame " << i << std::endl;
            std::cout << "  GT Position: " << position.transpose() << std::endl;
            std::cout << "  GT Orientation (quat): " 
                      << orientation.w() << " " 
                      << orientation.x() << " "
                      << orientation.y() << " "
                      << orientation.z() << std::endl;
            
            // 使用地面真值评估算法性能
            // ...
        }
    }
    
    return 0;
}
```

### 示例5：IMU预积分
```cpp
#include "dataset.h"
#include "imu.h"

int main() {
    Dataset dataset("/path/to/EuRoC/MH_03_medium", true);
    
    // 创建预积分对象
    Preinteration preint;
    preint.SetNoiseAndWalk(0.01, 0.1, 0.0001, 0.001);  // 设置噪声参数
    
    for (size_t i = 0; i < dataset.GetDatasetLength() - 1; ++i) {
        cv::Mat left_image, right_image;
        ImuDataList imu_data;
        double timestamp;
        
        if (!dataset.GetData(i, left_image, right_image, imu_data, timestamp)) {
            continue;
        }
        
        // 获取下一帧时间戳
        cv::Mat next_left, next_right;
        ImuDataList next_imu;
        double next_timestamp;
        dataset.GetData(i + 1, next_left, next_right, next_imu, next_timestamp);
        
        // 添加IMU数据进行预积分
        preint.AddBatchData(imu_data, timestamp, next_timestamp);
        
        std::cout << "Preintegration delta P: " 
                  << preint.dP.transpose() << std::endl;
        std::cout << "Preintegration delta V: "
                  << preint.dV.transpose() << std::endl;
    }
    
    return 0;
}
```

### 示例6：在ROS中使用
```cpp
#include "dataset.h"
#include "ros_publisher.h"
#include <ros/ros.h>

int main(int argc, char** argv) {
    ros::init(argc, argv, "euroc_player");
    ros::NodeHandle nh;
    
    // 创建数据集
    std::string dataroot;
    nh.param<std::string>("dataroot", dataroot, "");
    Dataset dataset(dataroot, true);
    
    // 创建ROS发布器
    RosPublisherConfig pub_config;
    pub_config.frame_pose = true;
    pub_config.feature = true;
    RosPublisher publisher(pub_config, nh);
    
    ros::Rate rate(20);  // 20 Hz
    
    for (size_t i = 0; i < dataset.GetDatasetLength() && ros::ok(); ++i) {
        cv::Mat left_image, right_image;
        ImuDataList imu_data;
        double timestamp;
        
        if (dataset.GetData(i, left_image, right_image, imu_data, timestamp)) {
            // 处理数据并发布
            // ...
            
            rate.sleep();
        }
    }
    
    return 0;
}
```

---

## 配置选项

### 标准配置
```cpp
// 双目 + IMU
EuRoCConfig config;
config.dataroot = "/path/to/dataset";
config.use_imu = true;
config.use_stereo = true;
config.use_ground_truth = false;
```

### 单目配置
```cpp
EuRoCConfig config;
config.dataroot = "/path/to/dataset";
config.use_imu = true;
config.use_stereo = false;  // 只使用cam0
```

### 评估模式配置
```cpp
EuRoCConfig config;
config.dataroot = "/path/to/dataset";
config.use_imu = true;
config.use_stereo = true;
config.use_ground_truth = true;  // 加载地面真值用于评估
```

### 自定义文件夹路径
```cpp
EuRoCConfig config;
config.dataroot = "/custom/path";
config.cam0_folder = "custom_cam0";
config.cam1_folder = "custom_cam1";
config.imu_folder = "custom_imu";
```

---

## 常见问题

### Q1: 如何处理时间戳？

**A:** EuRoC数据集使用纳秒级时间戳。Dataset类自动将其转换为秒级浮点数：
```cpp
double timestamp;  // 单位：秒
// 内部转换：timestamp = nanoseconds / 1e9
```

### Q2: IMU数据如何同步到图像？

**A:** Dataset类自动将两帧图像之间的IMU数据关联到后一帧：
```cpp
// 帧i-1到帧i之间的所有IMU数据会被关联到帧i
ImuDataList imu_data;  // 包含前一帧到当前帧之间的所有IMU测量
```

### Q3: 如何获取相机内参？

**A:** 相机内参存储在`sensor.yaml`文件中，需要单独解析：
```yaml
# cam0/sensor.yaml
intrinsics: [458.654, 457.296, 367.215, 248.375]  # [fx, fy, cx, cy]
distortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]
```

建议使用OpenCV的FileStorage读取：
```cpp
cv::FileStorage fs("cam0/sensor.yaml", cv::FileStorage::READ);
std::vector<double> intrinsics;
fs["intrinsics"] >> intrinsics;
```

### Q4: 地面真值的坐标系是什么？

**A:** 
- 位置和姿态在全局惯性坐标系（通常是Vicon房间坐标系或起始位置）
- 速度在全局坐标系
- IMU偏置在IMU传感器坐标系

### Q5: 如何处理大数据集？

**A:** 对于大数据集，可以：

1. 按批次处理：
```cpp
size_t batch_size = 100;
for (size_t start = 0; start < dataset.GetDatasetLength(); start += batch_size) {
    size_t end = std::min(start + batch_size, dataset.GetDatasetLength());
    // 处理 [start, end) 范围的数据
}
```

2. 使用多线程：
```cpp
#pragma omp parallel for
for (size_t i = 0; i < dataset.GetDatasetLength(); ++i) {
    // 并行处理每一帧
}
```

### Q6: 数据集序列推荐

- **入门学习**：V1_01_easy, MH_01_easy
- **中等难度**：V2_02_medium, MH_03_medium
- **困难场景**：V2_03_difficult, MH_05_difficult
- **纯旋转**：V1_03_difficult（适合测试IMU）

### Q7: 典型的相机和IMU频率

- **相机**：20 Hz（每帧间隔50ms）
- **IMU**：200 Hz（每次测量间隔5ms）
- 因此，相邻两帧之间通常有约10个IMU测量值

### Q8: 错误处理
```cpp
Dataset dataset(dataroot, use_imu);

// 检查数据集是否有效
if (dataset.GetDatasetLength() == 0) {
    std::cerr << "Failed to load dataset!" << std::endl;
    return -1;
}

// 验证数据完整性
if (!dataset.ValidateDataset()) {
    std::cerr << "Dataset validation failed!" << std::endl;
    // 继续或退出
}

// 获取数据时检查返回值
if (!dataset.GetData(i, left, right, imu, timestamp)) {
    std::cerr << "Failed to get data at index " << i << std::endl;
    continue;
}
```

---

## 数据集统计信息

| 序列 | 时长 | 轨迹长度 | 难度 | 特点 |
|------|------|----------|------|------|
| MH_01_easy | 182s | 80.6m | 简单 | 慢速，良好光照 |
| MH_02_easy | 150s | 73.8m | 简单 | 慢速，良好光照 |
| MH_03_medium | 132s | 130.9m | 中等 | 中速 |
| MH_04_difficult | 99s | 91.7m | 困难 | 快速运动 |
| MH_05_difficult | 111s | 97.6m | 困难 | 快速运动 |
| V1_01_easy | 143s | 58.4m | 简单 | 慢速 |
| V1_02_medium | 83s | 75.9m | 中等 | 中速 |
| V1_03_difficult | 105s | 79.4m | 困难 | 快速，纯旋转 |
| V2_01_easy | 112s | 36.5m | 简单 | 慢速 |
| V2_02_medium | 115s | 83.2m | 中等 | 中速 |
| V2_03_difficult | 115s | 86.1m | 困难 | 快速 |

---

## 参考资料

1. **官方论文**:
   - Burri, M., et al. "The EuRoC micro aerial vehicle datasets." IJRR 2016.

2. **官方网站**:
   - https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

3. **标定文件说明**:
   - https://github.com/ethz-asl/kalibr/wiki/yaml-formats

4. **相关工具**:
   - Kalibr: 相机-IMU标定工具
   - EVO: 轨迹评估工具

---

## 许可证

EuRoC数据集采用 Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License。

使用数据集时请引用原始论文。
