#include "transformer_net.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <utility>

#include "NvInferPlugin.h"

using namespace tensorrt_log;
using namespace tensorrt_buffer;

TransformerNet::TransformerNet(TransformerNetConfig& config) 
    : config_(config), 
      encoder_engine_(nullptr), 
      decoder_engine_(nullptr),
      resized_width_(config.input_size),
      resized_height_(config.input_size) {
  
  setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
  
  // 计算 patch 相关参数
  num_patches_ = (resized_width_ / config_.patch_size) * 
                 (resized_height_ / config_.patch_size);
  patch_dim_ = config_.patch_size * config_.patch_size * 1; // 灰度图
  sequence_length_ = num_patches_ + 1; // +1 for [CLS] token
  
  // 预分配内存
  encoder_features_.resize(sequence_length_ * config_.d_model);
  positional_encodings_.resize(sequence_length_ * config_.d_model);
}

bool TransformerNet::build() {
  if (deserialize_engine()) {
    // 创建执行上下文
    if (!encoder_context_) {
      encoder_context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(
          encoder_engine_->createExecutionContext());
      if (!encoder_context_) {
        return false;
      }
    }

    if (!decoder_context_) {
      decoder_context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(
          decoder_engine_->createExecutionContext());
      if (!decoder_context_) {
        return false;
      }
    }

    // 获取 binding indices
    image_input_index_ = encoder_engine_->getBindingIndex("image_input");
    encoder_output_index_ = encoder_engine_->getBindingIndex("encoder_output");
    positional_encoding_index_ = encoder_engine_->getBindingIndex("pos_encoding");
    
    decoder_query_index_ = decoder_engine_->getBindingIndex("query_input");
    encoder_memory_index_ = decoder_engine_->getBindingIndex("encoder_memory");
    keypoint_output_index_ = decoder_engine_->getBindingIndex("keypoint_output");
    line_output_index_ = decoder_engine_->getBindingIndex("line_output");
    junction_output_index_ = decoder_engine_->getBindingIndex("junction_output");
    descriptor_output_index_ = decoder_engine_->getBindingIndex("descriptor_output");
    
    return true;
  }

  // 构建 Encoder
  auto builder_encoder = TensorRTUniquePtr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
  if (!builder_encoder) {
    return false;
  }

  const auto explicit_batch = 1U << static_cast<uint32_t>(
      nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network_encoder = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(
      builder_encoder->createNetworkV2(explicit_batch));
  if (!network_encoder) {
    return false;
  }

  auto config_encoder = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(
      builder_encoder->createBuilderConfig());
  if (!config_encoder) {
    return false;
  }

  auto parser_encoder = TensorRTUniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network_encoder, gLogger.getTRTLogger()));
  if (!parser_encoder) {
    return false;
  }

  // 设置优化配置
  auto profile_encoder = builder_encoder->createOptimizationProfile();
  if (!profile_encoder) {
    return false;
  }
  
  // 动态输入尺寸
  profile_encoder->setDimensions("image_input", 
      nvinfer1::OptProfileSelector::kMIN, 
      nvinfer1::Dims4(1, 1, 256, 256));
  profile_encoder->setDimensions("image_input", 
      nvinfer1::OptProfileSelector::kOPT, 
      nvinfer1::Dims4(1, 1, 512, 512));
  profile_encoder->setDimensions("image_input", 
      nvinfer1::OptProfileSelector::kMAX, 
      nvinfer1::Dims4(1, 1, 1024, 1024));
  
  config_encoder->addOptimizationProfile(profile_encoder);

  // 构建 Encoder 网络
  if (!construct_encoder_network(builder_encoder, network_encoder, 
                                 config_encoder, parser_encoder)) {
    return false;
  }

  auto profile_stream_encoder = makeCudaStream();
  if (!profile_stream_encoder) {
    return false;
  }
  config_encoder->setProfileStream(*profile_stream_encoder);

  TensorRTUniquePtr<nvinfer1::IHostMemory> plan_encoder{
      builder_encoder->buildSerializedNetwork(*network_encoder, *config_encoder)};
  if (!plan_encoder) {
    return false;
  }

  TensorRTUniquePtr<nvinfer1::IRuntime> runtime_encoder{
      nvinfer1::createInferRuntime(gLogger.getTRTLogger())};
  if (!runtime_encoder) {
    return false;
  }

  encoder_engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime_encoder->deserializeCudaEngine(plan_encoder->data(), 
                                             plan_encoder->size()));
  if (!encoder_engine_) {
    return false;
  }

  encoder_context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(
      encoder_engine_->createExecutionContext());
  if (!encoder_context_) {
    return false;
  }

  // 构建 Decoder (类似的流程)
  auto builder_decoder = TensorRTUniquePtr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
  if (!builder_decoder) {
    return false;
  }

  auto network_decoder = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(
      builder_decoder->createNetworkV2(explicit_batch));
  if (!network_decoder) {
    return false;
  }

  auto config_decoder = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(
      builder_decoder->createBuilderConfig());
  if (!config_decoder) {
    return false;
  }

  auto parser_decoder = TensorRTUniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network_decoder, gLogger.getTRTLogger()));
  if (!parser_decoder) {
    return false;
  }

  auto profile_decoder = builder_decoder->createOptimizationProfile();
  if (!profile_decoder) {
    return false;
  }

  // Decoder 输入配置
  profile_decoder->setDimensions("query_input", 
      nvinfer1::OptProfileSelector::kMIN, 
      nvinfer1::Dims3(1, 100, config_.d_model));
  profile_decoder->setDimensions("query_input", 
      nvinfer1::OptProfileSelector::kOPT, 
      nvinfer1::Dims3(1, 300, config_.d_model));
  profile_decoder->setDimensions("query_input", 
      nvinfer1::OptProfileSelector::kMAX, 
      nvinfer1::Dims3(1, 500, config_.d_model));
  
  profile_decoder->setDimensions("encoder_memory", 
      nvinfer1::OptProfileSelector::kMIN, 
      nvinfer1::Dims3(1, 256, config_.d_model));
  profile_decoder->setDimensions("encoder_memory", 
      nvinfer1::OptProfileSelector::kOPT, 
      nvinfer1::Dims3(1, 1024, config_.d_model));
  profile_decoder->setDimensions("encoder_memory", 
      nvinfer1::OptProfileSelector::kMAX, 
      nvinfer1::Dims3(1, 4096, config_.d_model));

  config_decoder->addOptimizationProfile(profile_decoder);

  if (!construct_decoder_network(builder_decoder, network_decoder, 
                                 config_decoder, parser_decoder)) {
    return false;
  }

  auto profile_stream_decoder = makeCudaStream();
  if (!profile_stream_decoder) {
    return false;
  }
  config_decoder->setProfileStream(*profile_stream_decoder);

  TensorRTUniquePtr<nvinfer1::IHostMemory> plan_decoder{
      builder_decoder->buildSerializedNetwork(*network_decoder, *config_decoder)};
  if (!plan_decoder) {
    return false;
  }

  TensorRTUniquePtr<nvinfer1::IRuntime> runtime_decoder{
      nvinfer1::createInferRuntime(gLogger.getTRTLogger())};
  if (!runtime_decoder) {
    return false;
  }

  decoder_engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime_decoder->deserializeCudaEngine(plan_decoder->data(), 
                                             plan_decoder->size()));
  if (!decoder_engine_) {
    return false;
  }

  decoder_context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(
      decoder_engine_->createExecutionContext());
  if (!decoder_context_) {
    return false;
  }

  // 保存引擎
  save_engine();

  // 获取 binding indices
  image_input_index_ = encoder_engine_->getBindingIndex("image_input");
  encoder_output_index_ = encoder_engine_->getBindingIndex("encoder_output");
  
  decoder_query_index_ = decoder_engine_->getBindingIndex("query_input");
  encoder_memory_index_ = decoder_engine_->getBindingIndex("encoder_memory");
  keypoint_output_index_ = decoder_engine_->getBindingIndex("keypoint_output");
  line_output_index_ = decoder_engine_->getBindingIndex("line_output");
  junction_output_index_ = decoder_engine_->getBindingIndex("junction_output");
  descriptor_output_index_ = decoder_engine_->getBindingIndex("descriptor_output");

  return true;
}

bool TransformerNet::construct_encoder_network(
    TensorRTUniquePtr<nvinfer1::IBuilder> &builder, 
    TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
    TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config, 
    TensorRTUniquePtr<nvonnxparser::IParser> &parser) const {
  
  auto parsed = parser->parseFromFile(
      config_.transformer_encoder_onnx.c_str(), 
      static_cast<int>(gLogger.getReportableSeverity()));
  
  if (!parsed) {
    return false;
  }

  // 启用 FP16 精度以加速
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
  
  // 设置内存池大小
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1GB
  
  enableDLA(builder.get(), config.get(), -1);
  
  return true;
}

bool TransformerNet::construct_decoder_network(
    TensorRTUniquePtr<nvinfer1::IBuilder> &builder, 
    TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
    TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config, 
    TensorRTUniquePtr<nvonnxparser::IParser> &parser) const {
  
  auto parsed = parser->parseFromFile(
      config_.transformer_decoder_onnx.c_str(), 
      static_cast<int>(gLogger.getReportableSeverity()));
  
  if (!parsed) {
    return false;
  }

  config->setFlag(nvinfer1::BuilderFlag::kFP16);
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
  
  enableDLA(builder.get(), config.get(), -1);
  
  return true;
}

bool TransformerNet::infer(const cv::Mat &image, 
                           Eigen::Matrix<float, 259, Eigen::Dynamic> &features, 
                           std::vector<Eigen::Vector4d>& lines, 
                           Eigen::Matrix<float, 259, Eigen::Dynamic>& junctions, 
                           bool junction_detection) {
  
  // 设置 Encoder 输入维度
  encoder_context_->setBindingDimensions(image_input_index_, 
      nvinfer1::Dims4(1, 1, resized_height_, resized_width_));

  BufferManager encoder_buffers(encoder_engine_, 0, encoder_context_.get());

  // 图像预处理
  if (!preprocess_image(encoder_buffers, image)) {
    return false;
  }

  // 生成位置编码
  if (!generate_positional_encoding(sequence_length_, config_.d_model, 
                                     positional_encodings_)) {
    return false;
  }

  // 复制位置编码到缓冲区
  auto *pos_encoding_buffer = static_cast<float *>(
      encoder_buffers.getHostBuffer("pos_encoding"));
  std::copy(positional_encodings_.begin(), positional_encodings_.end(), 
            pos_encoding_buffer);

  encoder_buffers.copyInputToDevice();

  // 执行 Encoder
  bool status = encoder_context_->executeV2(encoder_buffers.getDeviceBindings().data());
  if (!status) {
    return false;
  }

  encoder_buffers.copyOutputToHost();

  // 获取 Encoder 输出
  auto *encoder_output = static_cast<float *>(
      encoder_buffers.getHostBuffer("encoder_output"));
  
  // 准备 Decoder 输入
  decoder_context_->setBindingDimensions(decoder_query_index_, 
      nvinfer1::Dims3(1, config_.max_keypoints + config_.max_lines, config_.d_model));
  decoder_context_->setBindingDimensions(encoder_memory_index_, 
      nvinfer1::Dims3(1, sequence_length_, config_.d_model));

  BufferManager decoder_buffers(decoder_engine_, 0, decoder_context_.get());

  // 初始化查询向量（可学习的嵌入）
  auto *query_buffer = static_cast<float *>(
      decoder_buffers.getHostBuffer("query_input"));
  // 这里应该使用预训练的查询嵌入，这里简化为随机初始化
  std::fill_n(query_buffer, 
              (config_.max_keypoints + config_.max_lines) * config_.d_model, 
              0.0f);

  // 复制 Encoder 输出作为 memory
  auto *memory_buffer = static_cast<float *>(
      decoder_buffers.getHostBuffer("encoder_memory"));
  std::copy(encoder_output, 
            encoder_output + sequence_length_ * config_.d_model, 
            memory_buffer);

  decoder_buffers.copyInputToDevice();

  // 执行 Decoder
  status = decoder_context_->executeV2(decoder_buffers.getDeviceBindings().data());
  if (!status) {
    return false;
  }

  decoder_buffers.copyOutputToHost();

  // 后处理输出
  if (!postprocess_output(decoder_buffers, features, lines, junctions, 
                          junction_detection)) {
    return false;
  }

  return true;
}

bool TransformerNet::preprocess_image(const BufferManager &buffers, 
                                      const cv::Mat &image) {
  if (image.empty()) {
    return false;
  }

  input_width_ = image.cols;
  input_height_ = image.rows;

  w_scale_ = static_cast<float>(input_width_) / resized_width_;
  h_scale_ = static_cast<float>(input_height_) / resized_height_;

  auto *host_data_buffer = static_cast<float *>(buffers.getHostBuffer("image_input"));

  cv::Mat resized_image;
  cv::resize(image, resized_image, cv::Size(resized_width_, resized_height_));

  // 归一化到 [0, 1]
  for (int row = 0; row < resized_height_; ++row) {
    const uchar *ptr = resized_image.ptr(row);
    int row_shift = row * resized_width_;
    for (int col = 0; col < resized_width_; ++col) {
      host_data_buffer[row_shift + col] = static_cast<float>(ptr[0]) / 255.0f;
      ptr++;
    }
  }

  return true;
}

bool TransformerNet::generate_positional_encoding(int sequence_length, 
                                                   int d_model, 
                                                   std::vector<float> &pos_encoding) {
  pos_encoding.resize(sequence_length * d_model);

  for (int pos = 0; pos < sequence_length; ++pos) {
    for (int i = 0; i < d_model; ++i) {
      float angle = pos / std::pow(10000.0f, 2.0f * i / d_model);
      if (i % 2 == 0) {
        pos_encoding[pos * d_model + i] = std::sin(angle);
      } else {
        pos_encoding[pos * d_model + i] = std::cos(angle);
      }
    }
  }

  return true;
}

bool TransformerNet::postprocess_output(
    const BufferManager &buffers, 
    Eigen::Matrix<float, 259, Eigen::Dynamic> &features, 
    std::vector<Eigen::Vector4d>& lines, 
    Eigen::Matrix<float, 259, Eigen::Dynamic>& junctions, 
    bool junction_detection) {
  
  auto *keypoint_logits = static_cast<float *>(
      buffers.getHostBuffer("keypoint_output"));
  auto *line_logits = static_cast<float *>(
      buffers.getHostBuffer("line_output"));
  auto *junction_logits = static_cast<float *>(
      buffers.getHostBuffer("junction_output"));
  auto *descriptors = static_cast<float *>(
      buffers.getHostBuffer("descriptor_output"));

  // 解码关键点
  if (!decode_keypoints(keypoint_logits, descriptors, features)) {
    return false;
  }

  // 解码线段
  if (!decode_lines(line_logits, lines)) {
    return false;
  }

  // 应用 NMS
  if (!apply_nms(lines, 0.5f)) {
    return false;
  }

  // 解码交点（如果需要）
  if (junction_detection) {
    if (!decode_junctions(junction_logits, descriptors, junctions)) {
      return false;
    }
    
    // 缩放到原始图像坐标
    junctions.block(1, 0, 1, junctions.cols()) = 
        junctions.block(1, 0, 1, junctions.cols()) * w_scale_;
    junctions.block(2, 0, 1, junctions.cols()) = 
        junctions.block(2, 0, 1, junctions.cols()) * h_scale_;
  }

  // 缩放关键点和线段到原始图像坐标
  features.block(1, 0, 1, features.cols()) = 
      features.block(1, 0, 1, features.cols()) * w_scale_;
  features.block(2, 0, 1, features.cols()) = 
      features.block(2, 0, 1, features.cols()) * h_scale_;

  for (auto &line : lines) {
    line[0] *= w_scale_;
    line[1] *= h_scale_;
    line[2] *= w_scale_;
    line[3] *= h_scale_;
  }

  return true;
}

bool TransformerNet::decode_keypoints(const float* keypoint_logits, 
                                      const float* descriptors, 
                                      Eigen::Matrix<float, 259, Eigen::Dynamic> &features) {
  std::vector<float> scores;
  std::vector<float> xs, ys;
  
  scores.reserve(config_.max_keypoints);
  xs.reserve(config_.max_keypoints);
  ys.reserve(config_.max_keypoints);

  // keypoint_logits 格式: [num_queries, 4] (score, x, y, scale)
  for (int i = 0; i < config_.max_keypoints; ++i) {
    float score = keypoint_logits[i * 4];
    
    if (score < config_.keypoint_threshold) {
      continue;
    }

    float x = keypoint_logits[i * 4 + 1] * resized_width_;
    float y = keypoint_logits[i * 4 + 2] * resized_height_;

    // 边界检查
    if (x < config_.remove_borders || x >= resized_width_ - config_.remove_borders ||
        y < config_.remove_borders || y >= resized_height_ - config_.remove_borders) {
      continue;
    }

    scores.push_back(score);
    xs.push_back(x);
    ys.push_back(y);
  }

  // 按分数排序
  if (scores.size() > config_.max_keypoints) {
    std::vector<size_t> indices = argsort(scores, true);
    scores.resize(config_.max_keypoints);
    xs.resize(config_.max_keypoints);
    ys.resize(config_.max_keypoints);
    
    for (int i = 0; i < config_.max_keypoints; ++i) {
      scores[i] = scores[indices[i]];
      xs[i] = xs[indices[i]];
      ys[i] = ys[indices[i]];
    }
  }

  // 构建特征矩阵
  int num_keypoints = scores.size();
  features.resize(259, num_keypoints);
  
  for (int i = 0; i < num_keypoints; ++i) {
    features(0, i) = scores[i];
    features(1, i) = xs[i];
    features(2, i) = ys[i];
  }

  // 提取描述符
  extract_descriptors_at_points(descriptors, features, 
                                resized_height_ / 8, resized_width_ / 8, 256);

  return true;
}

bool TransformerNet::decode_lines(const float* line_logits, 
                                  std::vector<Eigen::Vector4d>& lines) {
  lines.clear();
  
  const float length_threshold_sq = config_.line_length_threshold * 
                                    config_.line_length_threshold;

  // line_logits 格式: [num_queries, 5] (score, x1, y1, x2, y2)
  for (int i = 0; i < config_.max_lines; ++i) {
    float score = line_logits[i * 5];
    
    if (score < config_.line_threshold) {
      continue;
    }

    float x1 = line_logits[i * 5 + 1] * resized_width_;
    float y1 = line_logits[i * 5 + 2] * resized_height_;
    float x2 = line_logits[i * 5 + 3] * resized_width_;
    float y2 = line_logits[i * 5 + 4] * resized_height_;

    // 长度检查
    float length_sq = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
    if (length_sq < length_threshold_sq) {
      continue;
    }

    // 边界检查
    int border = config_.remove_borders;
    bool valid = (x1 > border && x1 < resized_width_ - border &&
                 y1 > border && y1 < resized_height_ - border &&
                 x2 > border && x2 < resized_width_ - border &&
                 y2 > border && y2 < resized_height_ - border);
    
    if (!valid) {
      continue;
    }

    lines.emplace_back(x1, y1, x2, y2);
  }

  return true;
}

bool TransformerNet::decode_junctions(const float* junction_logits, 
                                      const float* descriptors,
                                      Eigen::Matrix<float, 259, Eigen::Dynamic> &junctions) {
  std::vector<float> scores;
  std::vector<float> xs, ys;
  
  // junction_logits 格式: [H, W, 1]
  int border = std::max(config_.remove_borders, 0);
  
  for (int y = border; y < resized_height_ - border; ++y) {
    for (int x = border; x < resized_width_ - border; ++x) {
      int idx = y * resized_width_ + x;
      float score = junction_logits[idx];
      
      if (score > 0.5f) { // Junction threshold
        scores.push_back(score);
        xs.push_back(static_cast<float>(x));
        ys.push_back(static_cast<float>(y));
      }
    }
  }

  int num_junctions = scores.size();
  junctions.resize(259, num_junctions);
  
  for (int i = 0; i < num_junctions; ++i) {
    junctions(0, i) = scores[i];
    junctions(1, i) = xs[i];
    junctions(2, i) = ys[i];
  }

  // 提取描述符
  extract_descriptors_at_points(descriptors, junctions, 
                                resized_height_ / 8, resized_width_ / 8, 256);

  return true;
}

bool TransformerNet::apply_nms(std::vector<Eigen::Vector4d>& lines, 
                               float iou_threshold) {
  if (lines.empty()) {
    return true;
  }

  // 计算每条线的长度作为分数
  std::vector<float> scores;
  for (const auto& line : lines) {
    float length = std::sqrt((line[2] - line[0]) * (line[2] - line[0]) +
                            (line[3] - line[1]) * (line[3] - line[1]));
    scores.push_back(length);
  }

  std::vector<size_t> indices = argsort(scores, true);
  std::vector<bool> keep(lines.size(), true);

  for (size_t i = 0; i < indices.size(); ++i) {
    if (!keep[indices[i]]) {
      continue;
    }

    for (size_t j = i + 1; j < indices.size(); ++j) {
      if (!keep[indices[j]]) {
        continue;
      }

      float iou = compute_line_iou(lines[indices[i]], lines[indices[j]]);
      if (iou > iou_threshold) {
        keep[indices[j]] = false;
      }
    }
  }

  // 过滤线段
  std::vector<Eigen::Vector4d> filtered_lines;
  for (size_t i = 0; i < lines.size(); ++i) {
    if (keep[i]) {
      filtered_lines.push_back(lines[i]);
    }
  }

  lines = std::move(filtered_lines);
  return true;
}

void TransformerNet::extract_descriptors_at_points(
    const float *descriptor_map, 
    Eigen::Matrix<float, 259, Eigen::Dynamic> &features, 
    int h, int w, int descriptor_dim) {
  
  // 使用双线性插值提取描述符
  for (int j = 0; j < features.cols(); ++j) {
    float x = features(1, j);
    float y = features(2, j);

    // 归一化到特征图坐标
    float fx = x / resized_width_ * w;
    float fy = y / resized_height_ * h;

    int x0 = clip(static_cast<int>(std::floor(fx)), w);
    int y0 = clip(static_cast<int>(std::floor(fy)), h);
    int x1 = clip(x0 + 1, w);
    int y1 = clip(y0 + 1, h);

    float wx = fx - x0;
    float wy = fy - y0;

    for (int d = 0; d < descriptor_dim; ++d) {
      float v00 = descriptor_map[d * h * w + y0 * w + x0];
      float v01 = descriptor_map[d * h * w + y0 * w + x1];
      float v10 = descriptor_map[d * h * w + y1 * w + x0];
      float v11 = descriptor_map[d * h * w + y1 * w + x1];

      float value = (1 - wx) * (1 - wy) * v00 +
                    wx * (1 - wy) * v01 +
                    (1 - wx) * wy * v10 +
                    wx * wy * v11;

      features(d + 3, j) = value;
    }
  }

  // L2 归一化描述符
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> descriptors = 
      features.block(3, 0, descriptor_dim, features.cols());
  descriptors.colwise().normalize();
  features.block(3, 0, descriptor_dim, features.cols()) = descriptors;
}

std::vector<size_t> TransformerNet::argsort(const std::vector<float> &scores, 
                                            bool descending) {
  std::vector<size_t> indices(scores.size());
  std::iota(indices.begin(), indices.end(), 0);
  
  if (descending) {
    std::sort(indices.begin(), indices.end(), 
              [&scores](size_t i1, size_t i2) { return scores[i1] > scores[i2]; });
  } else {
    std::sort(indices.begin(), indices.end(), 
              [&scores](size_t i1, size_t i2) { return scores[i1] < scores[i2]; });
  }
  
  return indices;
}

int TransformerNet::clip(int val, int max) {
  return std::max(0, std::min(val, max - 1));
}

float TransformerNet::compute_line_iou(const Eigen::Vector4d& line1, 
                                       const Eigen::Vector4d& line2) {
  // 简化的线段 IoU 计算
  // 这里使用端点距离作为近似
  float dist_start = std::sqrt((line1[0] - line2[0]) * (line1[0] - line2[0]) +
                               (line1[1] - line2[1]) * (line1[1] - line2[1]));
  float dist_end = std::sqrt((line1[2] - line2[2]) * (line1[2] - line2[2]) +
                             (line1[3] - line2[3]) * (line1[3] - line2[3]));
  
  float length1 = std::sqrt((line1[2] - line1[0]) * (line1[2] - line1[0]) +
                           (line1[3] - line1[1]) * (line1[3] - line1[1]));
  float length2 = std::sqrt((line2[2] - line2[0]) * (line2[2] - line2[0]) +
                           (line2[3] - line2[1]) * (line2[3] - line2[1]));
  
  float avg_length = (length1 + length2) / 2.0f;
  float avg_dist = (dist_start + dist_end) / 2.0f;
  
  return 1.0f - std::min(avg_dist / avg_length, 1.0f);
}

void TransformerNet::save_engine() {
  if (encoder_engine_) {
    nvinfer1::IHostMemory *data = encoder_engine_->serialize();
    std::ofstream file(config_.transformer_encoder_engine, std::ios::binary);
    if (file) {
      file.write(reinterpret_cast<const char *>(data->data()), data->size());
    }
    data->destroy();
  }

  if (decoder_engine_) {
    nvinfer1::IHostMemory *data = decoder_engine_->serialize();
    std::ofstream file(config_.transformer_decoder_engine, std::ios::binary);
    if (file) {
      file.write(reinterpret_cast<const char *>(data->data()), data->size());
    }
    data->destroy();
  }
}

bool TransformerNet::deserialize_engine() {
  std::ifstream encoder_file(config_.transformer_encoder_engine, std::ios::binary);
  std::ifstream decoder_file(config_.transformer_decoder_engine, std::ios::binary);
  
  bool encoder_loaded = false;
  bool decoder_loaded = false;

  if (encoder_file.is_open()) {
    encoder_file.seekg(0, std::ifstream::end);
    size_t size = encoder_file.tellg();
    encoder_file.seekg(0, std::ifstream::beg);
    
    char *model_stream = new char[size];
    encoder_file.read(model_stream, size);
    encoder_file.close();
    
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
    if (runtime) {
      encoder_engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
          runtime->deserializeCudaEngine(model_stream, size));
      encoder_loaded = (encoder_engine_ != nullptr);
      runtime->destroy();
    }
    
    delete[] model_stream;
  }

  if (decoder_file.is_open()) {
    decoder_file.seekg(0, std::ifstream::end);
    size_t size = decoder_file.tellg();
    decoder_file.seekg(0, std::ifstream::beg);
    
    char *model_stream = new char[size];
    decoder_file.read(model_stream, size);
    decoder_file.close();
    
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
    if (runtime) {
      decoder_engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
          runtime->deserializeCudaEngine(model_stream, size));
      decoder_loaded = (decoder_engine_ != nullptr);
      runtime->destroy();
    }
    
    delete[] model_stream;
  }

  return encoder_loaded && decoder_loaded;
}

bool TransformerNet::visualize_attention(const float* attention_weights, 
                                        int num_heads, 
                                        int seq_len,
                                        const std::string& save_path) {
  // 注意力可视化的实现（调试用）
  // 这里省略具体实现
  return true;
}
