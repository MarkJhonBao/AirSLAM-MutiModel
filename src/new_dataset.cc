#include "dataset.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cmath>

#include "utils.h"

Dataset::Dataset(const EuRoCConfig& config) 
    : _config(config),
      _use_imu(config.use_imu),
      _use_stereo(config.use_stereo),
      _use_ground_truth(config.use_ground_truth) {
  
  // 检查数据根目录
  if (!PathExists(_config.dataroot)) {
    std::cout << "Error: Dataset root path does not exist: " 
              << _config.dataroot << std::endl;
    exit(EXIT_FAILURE);
  }
  
  // 构建数据路径
  _cam0_folder_path = ConcatenateFolderAndFileName(_config.dataroot, _config.cam0_folder);
  _cam1_folder_path = ConcatenateFolderAndFileName(_config.dataroot, _config.cam1_folder);
  _imu_file_path = ConcatenateFolderAndFileName(
      ConcatenateFolderAndFileName(_config.dataroot, _config.imu_folder),
      _config.imu_data_csv);
  _ground_truth_file_path = ConcatenateFolderAndFileName(
      ConcatenateFolderAndFileName(_config.dataroot, _config.ground_truth_folder),
      _config.ground_truth_csv);
  
  // 加载相机数据
  std::string cam0_csv = ConcatenateFolderAndFileName(_cam0_folder_path, _config.cam0_data_csv);
  std::string cam0_data_folder = ConcatenateFolderAndFileName(_cam0_folder_path, 
                                                               _config.cam0_data_folder);
  LoadCameraData(cam0_csv, cam0_data_folder, _cam0_data);
  
  if (_use_stereo) {
    std::string cam1_csv = ConcatenateFolderAndFileName(_cam1_folder_path, _config.cam1_data_csv);
    std::string cam1_data_folder = ConcatenateFolderAndFileName(_cam1_folder_path, 
                                                                 _config.cam1_data_folder);
    LoadCameraData(cam1_csv, cam1_data_folder, _cam1_data);
  }
  
  // 加载IMU数据
  if (_use_imu) {
    if (FileExists(_imu_file_path)) {
      LoadImuData(_imu_file_path, _all_imu_data);
      std::cout << "Loaded " << _all_imu_data.size() << " IMU measurements" << std::endl;
    } else {
      std::cout << "Warning: IMU file not found: " << _imu_file_path << std::endl;
      _use_imu = false;
    }
  }
  
  // 加载地面真值数据
  if (_use_ground_truth) {
    if (FileExists(_ground_truth_file_path)) {
      LoadGroundTruthData(_ground_truth_file_path, _ground_truth_data);
      std::cout << "Loaded " << _ground_truth_data.size() 
                << " ground truth poses" << std::endl;
    } else {
      std::cout << "Warning: Ground truth file not found: " 
                << _ground_truth_file_path << std::endl;
      _use_ground_truth = false;
    }
  }
  
  // 同步数据
  SynchronizeData();
  
  // 打印数据集信息
  PrintDatasetInfo();
}

Dataset::Dataset(const std::string& dataroot, bool use_imu) 
    : _use_imu(use_imu), _use_stereo(true), _use_ground_truth(false) {
  
  _config.dataroot = dataroot;
  _config.use_imu = use_imu;
  _config.use_stereo = true;
  _config.use_ground_truth = false;
  
  // 使用默认配置初始化
  Dataset(EuRoCConfig(_config));
}

void Dataset::LoadCameraData(const std::string& csv_path, 
                             const std::string& data_folder,
                             CameraDataList& camera_data) {
  if (!FileExists(csv_path)) {
    std::cout << "Error: Camera CSV file not found: " << csv_path << std::endl;
    exit(EXIT_FAILURE);
  }
  
  std::vector<std::vector<std::string>> lines;
  ReadTxt(csv_path, lines, ",");
  
  if (lines.empty()) {
    std::cout << "Error: Empty camera CSV file: " << csv_path << std::endl;
    exit(EXIT_FAILURE);
  }
  
  // 跳过标题行
  camera_data.reserve(lines.size() - 1);
  for (size_t i = 1; i < lines.size(); ++i) {
    if (lines[i].size() < 2) continue;
    
    double timestamp = ParseTimestamp(lines[i][0]);
    std::string filename = ConcatenateFolderAndFileName(data_folder, lines[i][1]);
    
    camera_data.emplace_back(timestamp, filename);
  }
  
  std::cout << "Loaded " << camera_data.size() << " camera frames from " 
            << csv_path << std::endl;
}

void Dataset::LoadImuData(const std::string& csv_path, ImuDataList& imu_data) {
  if (!FileExists(csv_path)) {
    std::cout << "Error: IMU CSV file not found: " << csv_path << std::endl;
    return;
  }
  
  std::vector<std::vector<std::string>> lines;
  ReadTxt(csv_path, lines, ",");
  
  if (lines.empty()) {
    std::cout << "Error: Empty IMU CSV file: " << csv_path << std::endl;
    return;
  }
  
  // 跳过标题行
  // timestamp [ns], w_RS_S_x [rad s^-1], w_RS_S_y [rad s^-1], w_RS_S_z [rad s^-1],
  // a_RS_S_x [m s^-2], a_RS_S_y [m s^-2], a_RS_S_z [m s^-2]
  imu_data.reserve(lines.size() - 1);
  for (size_t i = 1; i < lines.size(); ++i) {
    if (lines[i].size() < 7) continue;
    
    ImuData data;
    data.timestamp = ParseTimestamp(lines[i][0]);
    data.gyr << std::stod(lines[i][1]), std::stod(lines[i][2]), std::stod(lines[i][3]);
    data.acc << std::stod(lines[i][4]), std::stod(lines[i][5]), std::stod(lines[i][6]);
    
    imu_data.push_back(data);
  }
}

void Dataset::LoadGroundTruthData(const std::string& csv_path, 
                                  GroundTruthDataList& ground_truth_data) {
  if (!FileExists(csv_path)) {
    std::cout << "Warning: Ground truth CSV file not found: " << csv_path << std::endl;
    return;
  }
  
  std::vector<std::vector<std::string>> lines;
  ReadTxt(csv_path, lines, ",");
  
  if (lines.empty()) {
    std::cout << "Warning: Empty ground truth CSV file: " << csv_path << std::endl;
    return;
  }
  
  // 跳过标题行
  // timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m],
  // q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [],
  // v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1],
  // b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1],
  // b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]
  ground_truth_data.reserve(lines.size() - 1);
  for (size_t i = 1; i < lines.size(); ++i) {
    if (lines[i].size() < 17) continue;
    
    GroundTruthData data;
    data.timestamp = ParseTimestamp(lines[i][0]);
    
    // Position
    data.position << std::stod(lines[i][1]), std::stod(lines[i][2]), std::stod(lines[i][3]);
    
    // Orientation (quaternion: w, x, y, z)
    data.orientation.w() = std::stod(lines[i][4]);
    data.orientation.x() = std::stod(lines[i][5]);
    data.orientation.y() = std::stod(lines[i][6]);
    data.orientation.z() = std::stod(lines[i][7]);
    data.orientation.normalize();
    
    // Velocity
    data.velocity << std::stod(lines[i][8]), std::stod(lines[i][9]), std::stod(lines[i][10]);
    
    // Biases
    data.bias_gyr << std::stod(lines[i][11]), std::stod(lines[i][12]), std::stod(lines[i][13]);
    data.bias_acc << std::stod(lines[i][14]), std::stod(lines[i][15]), std::stod(lines[i][16]);
    
    ground_truth_data.push_back(data);
  }
}

void Dataset::SynchronizeData() {
  if (_cam0_data.empty()) {
    std::cout << "Error: No camera data loaded!" << std::endl;
    return;
  }
  
  size_t num_frames = _cam0_data.size();
  
  // 如果使用双目，检查相机数据是否匹配
  if (_use_stereo && _cam1_data.size() != _cam0_data.size()) {
    std::cout << "Warning: Cam0 and Cam1 have different number of frames. "
              << "Using minimum." << std::endl;
    num_frames = std::min(_cam0_data.size(), _cam1_data.size());
  }
  
  _timestamps.clear();
  _cam0_indices.clear();
  _cam1_indices.clear();
  
  // 获取时间范围
  double start_time = _cam0_data[0].timestamp;
  double end_time = _cam0_data[num_frames - 1].timestamp;
  
  // 如果有IMU数据，调整时间范围
  if (_use_imu && !_all_imu_data.empty()) {
    start_time = std::max(start_time, _all_imu_data.front().timestamp);
    end_time = std::min(end_time, _all_imu_data.back().timestamp);
  }
  
  // 同步相机帧
  for (size_t i = 0; i < num_frames; ++i) {
    double cam0_time = _cam0_data[i].timestamp;
    
    // 检查时间范围
    if (cam0_time < start_time || cam0_time > end_time) continue;
    
    // 如果使用双目，检查cam1时间戳是否接近
    if (_use_stereo) {
      double cam1_time = _cam1_data[i].timestamp;
      double time_diff = std::abs(cam0_time - cam1_time);
      if (time_diff > 0.001) { // 1ms tolerance
        std::cout << "Warning: Large timestamp difference between stereo cameras: "
                  << time_diff << " s at index " << i << std::endl;
      }
    }
    
    _timestamps.push_back(cam0_time);
    _cam0_indices.push_back(i);
    if (_use_stereo) {
      _cam1_indices.push_back(i);
    }
  }
  
  std::cout << "Synchronized " << _timestamps.size() << " frames" << std::endl;
  
  // 关联IMU数据
  if (_use_imu) {
    AssociateImuData();
  }
  
  // 同步地面真值
  if (_use_ground_truth && !_ground_truth_data.empty()) {
    _synchronized_ground_truth.resize(_timestamps.size());
    for (size_t i = 0; i < _timestamps.size(); ++i) {
      GetGroundTruthAtTime(_timestamps[i], _synchronized_ground_truth[i]);
    }
  }
}

void Dataset::AssociateImuData() {
  if (_all_imu_data.empty()) return;
  
  _synchronized_imu_data.clear();
  _synchronized_imu_data.resize(_timestamps.size());
  
  size_t imu_idx = 0;
  double last_image_time = -1.0;
  
  for (size_t i = 0; i < _timestamps.size(); ++i) {
    double image_time = _timestamps[i];
    ImuDataList batch_imu;
    
    // 收集两帧之间的所有IMU数据
    while (imu_idx < _all_imu_data.size()) {
      const ImuData& imu = _all_imu_data[imu_idx];
      
      // 跳过旧的IMU数据
      if (imu.timestamp < last_image_time) {
        imu_idx++;
        continue;
      }
      
      // 收集当前帧之前的IMU数据
      if (imu.timestamp <= image_time) {
        batch_imu.push_back(imu);
        imu_idx++;
      } else {
        break;
      }
    }
    
    // 回退一个，以便下一帧可以使用
    if (imu_idx > 0) imu_idx--;
    
    _synchronized_imu_data[i] = batch_imu;
    last_image_time = image_time;
  }
}

bool Dataset::GetData(size_t idx, cv::Mat& left_image, cv::Mat& right_image, 
                      ImuDataList& batch_imu_data, double& timestamp) {
  batch_imu_data.clear();
  
  if (idx >= _timestamps.size()) {
    return false;
  }
  
  // 读取左图
  size_t cam0_idx = _cam0_indices[idx];
  if (!LoadImage(_cam0_data[cam0_idx].image_filename, left_image)) {
    return false;
  }
  
  // 读取右图
  if (_use_stereo) {
    size_t cam1_idx = _cam1_indices[idx];
    if (!LoadImage(_cam1_data[cam1_idx].image_filename, right_image)) {
      return false;
    }
  }
  
  // 获取时间戳
  timestamp = _timestamps[idx];
  
  // 获取IMU数据
  if (_use_imu && idx < _synchronized_imu_data.size()) {
    batch_imu_data = _synchronized_imu_data[idx];
  }
  
  return true;
}

bool Dataset::GetData(size_t idx, cv::Mat& left_image, cv::Mat& right_image, 
                      ImuDataList& batch_imu_data, double& timestamp,
                      GroundTruthData& ground_truth) {
  if (!GetData(idx, left_image, right_image, batch_imu_data, timestamp)) {
    return false;
  }
  
  // 获取地面真值
  if (_use_ground_truth && idx < _synchronized_ground_truth.size()) {
    ground_truth = _synchronized_ground_truth[idx];
  }
  
  return true;
}

bool Dataset::GetMonoData(size_t idx, cv::Mat& image, 
                          ImuDataList& batch_imu_data, double& timestamp) {
  cv::Mat dummy_image;
  return GetData(idx, image, dummy_image, batch_imu_data, timestamp);
}

bool Dataset::LoadImage(const std::string& image_path, cv::Mat& image) const {
  if (!FileExists(image_path)) {
    std::cout << "Error: Image file not found: " << image_path << std::endl;
    return false;
  }
  
  image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
  if (image.empty()) {
    std::cout << "Error: Failed to load image: " << image_path << std::endl;
    return false;
  }
  
  return true;
}

double Dataset::ParseTimestamp(const std::string& timestamp_str) const {
  // EuRoC时间戳格式：纳秒级整数
  long long timestamp_ns = std::stoll(timestamp_str);
  return static_cast<double>(timestamp_ns) / 1e9;
}

bool Dataset::GetGroundTruthAtTime(double timestamp, GroundTruthData& gt_data) const {
  if (_ground_truth_data.empty()) {
    return false;
  }
  
  // 线性插值查找最近的地面真值
  auto it = std::lower_bound(_ground_truth_data.begin(), _ground_truth_data.end(),
                             timestamp,
                             [](const GroundTruthData& data, double t) {
                               return data.timestamp < t;
                             });
  
  if (it == _ground_truth_data.end()) {
    gt_data = _ground_truth_data.back();
    return true;
  }
  
  if (it == _ground_truth_data.begin()) {
    gt_data = _ground_truth_data.front();
    return true;
  }
  
  // 线性插值
  auto it_prev = it - 1;
  double t1 = it_prev->timestamp;
  double t2 = it->timestamp;
  double alpha = (timestamp - t1) / (t2 - t1);
  
  gt_data.timestamp = timestamp;
  gt_data.position = (1.0 - alpha) * it_prev->position + alpha * it->position;
  gt_data.velocity = (1.0 - alpha) * it_prev->velocity + alpha * it->velocity;
  gt_data.bias_gyr = (1.0 - alpha) * it_prev->bias_gyr + alpha * it->bias_gyr;
  gt_data.bias_acc = (1.0 - alpha) * it_prev->bias_acc + alpha * it->bias_acc;
  
  // 四元数球面线性插值 (SLERP)
  gt_data.orientation = it_prev->orientation.slerp(alpha, it->orientation);
  
  return true;
}

size_t Dataset::GetDatasetLength() const {
  return _timestamps.size();
}

size_t Dataset::GetImuDataLength() const {
  return _all_imu_data.size();
}

size_t Dataset::GetGroundTruthLength() const {
  return _ground_truth_data.size();
}

double Dataset::GetStartTime() const {
  return _timestamps.empty() ? 0.0 : _timestamps.front();
}

double Dataset::GetEndTime() const {
  return _timestamps.empty() ? 0.0 : _timestamps.back();
}

void Dataset::PrintDatasetInfo() const {
  std::cout << "\n========== Dataset Information ==========" << std::endl;
  std::cout << "Dataset root: " << _config.dataroot << std::endl;
  std::cout << "Number of synchronized frames: " << _timestamps.size() << std::endl;
  
  if (!_timestamps.empty()) {
    std::cout << "Time range: " << std::fixed << std::setprecision(6)
              << GetStartTime() << " - " << GetEndTime() 
              << " (" << (GetEndTime() - GetStartTime()) << " s)" << std::endl;
  }
  
  std::cout << "Stereo: " << (_use_stereo ? "Yes" : "No") << std::endl;
  
  if (_use_imu) {
    std::cout << "IMU measurements: " << _all_imu_data.size() << std::endl;
    if (!_all_imu_data.empty()) {
      double imu_rate = static_cast<double>(_all_imu_data.size()) / 
                       (GetEndTime() - GetStartTime());
      std::cout << "IMU rate: " << std::fixed << std::setprecision(1) 
                << imu_rate << " Hz" << std::endl;
    }
  }
  
  if (_use_ground_truth) {
    std::cout << "Ground truth poses: " << _ground_truth_data.size() << std::endl;
  }
  
  std::cout << "========================================\n" << std::endl;
}

bool Dataset::ValidateDataset() const {
  bool valid = true;
  
  // 检查基本数据
  if (_cam0_data.empty()) {
    std::cout << "Validation Error: No camera data loaded" << std::endl;
    valid = false;
  }
  
  if (_use_stereo && _cam1_data.empty()) {
    std::cout << "Validation Error: Stereo mode enabled but no cam1 data" << std::endl;
    valid = false;
  }
  
  if (_use_imu && _all_imu_data.empty()) {
    std::cout << "Validation Error: IMU mode enabled but no IMU data" << std::endl;
    valid = false;
  }
  
  // 检查时间戳单调性
  for (size_t i = 1; i < _timestamps.size(); ++i) {
    if (_timestamps[i] <= _timestamps[i-1]) {
      std::cout << "Validation Error: Non-monotonic timestamps at index " << i << std::endl;
      valid = false;
      break;
    }
  }
  
  // 检查图像文件
  size_t missing_images = 0;
  for (size_t i = 0; i < std::min(_cam0_indices.size(), size_t(10)); ++i) {
    if (!FileExists(_cam0_data[_cam0_indices[i]].image_filename)) {
      missing_images++;
    }
  }
  if (missing_images > 0) {
    std::cout << "Validation Warning: Found " << missing_images 
              << " missing images (checked first 10)" << std::endl;
  }
  
  if (valid) {
    std::cout << "Dataset validation passed!" << std::endl;
  }
  
  return valid;
}
