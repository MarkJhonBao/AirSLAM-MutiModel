#ifndef DATASET_H_
#define DATASET_H_

#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "imu.h"
#include "utils.h"

// EuRoC数据集配置
struct EuRoCConfig {
  std::string dataroot;
  bool use_imu = true;
  bool use_stereo = true;
  bool use_ground_truth = false;
  
  // 传感器配置
  std::string cam0_folder = "mav0/cam0";
  std::string cam1_folder = "mav0/cam1";
  std::string imu_folder = "mav0/imu0";
  std::string ground_truth_folder = "mav0/state_groundtruth_estimate0";
  
  // 数据文件
  std::string cam0_data_csv = "data.csv";
  std::string cam1_data_csv = "data.csv";
  std::string imu_data_csv = "data.csv";
  std::string ground_truth_csv = "data.csv";
  
  // 图像数据文件夹
  std::string cam0_data_folder = "data";
  std::string cam1_data_folder = "data";
};

// 地面真值数据结构
struct GroundTruthData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  double timestamp;
  Eigen::Vector3d position;      // p_RS_R_x, p_RS_R_y, p_RS_R_z
  Eigen::Quaterniond orientation; // q_RS_w, q_RS_x, q_RS_y, q_RS_z
  Eigen::Vector3d velocity;      // v_RS_R_x, v_RS_R_y, v_RS_R_z
  Eigen::Vector3d bias_gyr;      // b_w_RS_S_x, b_w_RS_S_y, b_w_RS_S_z
  Eigen::Vector3d bias_acc;      // b_a_RS_S_x, b_a_RS_S_y, b_a_RS_S_z
  
  GroundTruthData() : timestamp(0.0) {
    position.setZero();
    orientation.setIdentity();
    velocity.setZero();
    bias_gyr.setZero();
    bias_acc.setZero();
  }
  
  GroundTruthData& operator=(const GroundTruthData& other) {
    timestamp = other.timestamp;
    position = other.position;
    orientation = other.orientation;
    velocity = other.velocity;
    bias_gyr = other.bias_gyr;
    bias_acc = other.bias_acc;
    return *this;
  }
  
  Eigen::Matrix4d GetPose() const {
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3, 3>(0, 0) = orientation.toRotationMatrix();
    pose.block<3, 1>(0, 3) = position;
    return pose;
  }
};
typedef std::vector<GroundTruthData> GroundTruthDataList;

// 相机数据结构
struct CameraData {
  double timestamp;
  std::string image_filename;
  
  CameraData() : timestamp(0.0) {}
  CameraData(double t, const std::string& filename) 
    : timestamp(t), image_filename(filename) {}
};
typedef std::vector<CameraData> CameraDataList;

class Dataset {
public:
  // 构造函数
  Dataset(const EuRoCConfig& config);
  Dataset(const std::string& dataroot, bool use_imu = true);
  
  // 数据读取接口
  bool GetData(size_t idx, cv::Mat& left_image, cv::Mat& right_image, 
               ImuDataList& batch_imu_data, double& timestamp);
  
  bool GetData(size_t idx, cv::Mat& left_image, cv::Mat& right_image, 
               ImuDataList& batch_imu_data, double& timestamp,
               GroundTruthData& ground_truth);
  
  // 单目数据读取
  bool GetMonoData(size_t idx, cv::Mat& image, 
                   ImuDataList& batch_imu_data, double& timestamp);
  
  // 获取数据集信息
  size_t GetDatasetLength() const;
  size_t GetImuDataLength() const;
  size_t GetGroundTruthLength() const;
  
  // 获取时间范围
  double GetStartTime() const;
  double GetEndTime() const;
  
  // 地面真值相关
  bool HasGroundTruth() const { return _use_ground_truth && !_ground_truth_data.empty(); }
  bool GetGroundTruthAtTime(double timestamp, GroundTruthData& gt_data) const;
  const GroundTruthDataList& GetAllGroundTruth() const { return _ground_truth_data; }
  
  // IMU数据相关
  bool HasImu() const { return _use_imu && !_all_imu_data.empty(); }
  const ImuDataList& GetAllImuData() const { return _all_imu_data; }
  
  // 相机数据相关
  bool IsStereo() const { return _use_stereo; }
  const CameraDataList& GetCam0Data() const { return _cam0_data; }
  const CameraDataList& GetCam1Data() const { return _cam1_data; }
  
  // 数据集统计
  void PrintDatasetInfo() const;
  
  // 数据验证
  bool ValidateDataset() const;

private:
  // 数据加载函数
  void LoadCameraData(const std::string& csv_path, const std::string& data_folder,
                      CameraDataList& camera_data);
  void LoadImuData(const std::string& csv_path, ImuDataList& imu_data);
  void LoadGroundTruthData(const std::string& csv_path, 
                           GroundTruthDataList& ground_truth_data);
  
  // 数据同步
  void SynchronizeData();
  void AssociateImuData();
  
  // 工具函数
  bool LoadImage(const std::string& image_path, cv::Mat& image) const;
  double ParseTimestamp(const std::string& timestamp_str) const;
  
  // 配置
  EuRoCConfig _config;
  bool _use_imu;
  bool _use_stereo;
  bool _use_ground_truth;
  
  // 原始数据
  CameraDataList _cam0_data;
  CameraDataList _cam1_data;
  ImuDataList _all_imu_data;
  GroundTruthDataList _ground_truth_data;
  
  // 同步后的数据
  std::vector<size_t> _cam0_indices;
  std::vector<size_t> _cam1_indices;
  std::vector<double> _timestamps;
  std::vector<ImuDataList> _synchronized_imu_data;
  std::vector<GroundTruthData> _synchronized_ground_truth;
  
  // 数据路径
  std::string _cam0_folder_path;
  std::string _cam1_folder_path;
  std::string _imu_file_path;
  std::string _ground_truth_file_path;
};

typedef std::shared_ptr<Dataset> DatasetPtr;

#endif // DATASET_H_
