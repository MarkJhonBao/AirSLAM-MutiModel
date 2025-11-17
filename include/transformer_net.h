#ifndef PLNET_TRANSFORMER_NET_H
#define PLNET_TRANSFORMER_NET_H

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <Eigen/Core>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "3rdparty/tensorrtbuffer/include/buffers.h"
#include "read_configs.h"

using tensorrt_buffer::TensorRTUniquePtr;

// Transformer 配置结构
struct TransformerNetConfig {
  std::string transformer_encoder_onnx;
  std::string transformer_decoder_onnx;
  std::string transformer_encoder_engine;
  std::string transformer_decoder_engine;
  
  // Transformer 特定参数
  int num_encoder_layers = 6;
  int num_decoder_layers = 6;
  int d_model = 256;
  int num_heads = 8;
  int dim_feedforward = 2048;
  float dropout = 0.1;
  
  // 检测参数
  float keypoint_threshold = 0.015;
  float line_threshold = 0.1;
  float line_length_threshold = 15.0;
  int max_keypoints = 300;
  int max_lines = 500;
  int remove_borders = 4;
  
  // 图像参数
  int input_size = 512;
  int patch_size = 16;
};

class TransformerNet {
 public:
  explicit TransformerNet(TransformerNetConfig& config);
  ~TransformerNet() = default;

  // 构建 TensorRT 引擎
  bool build();

  // 主推理接口
  bool infer(const cv::Mat &image, 
             Eigen::Matrix<float, 259, Eigen::Dynamic> &features, 
             std::vector<Eigen::Vector4d>& lines, 
             Eigen::Matrix<float, 259, Eigen::Dynamic>& junctions, 
             bool junction_detection = false);

  // 序列化和反序列化引擎
  void save_engine();
  bool deserialize_engine();

 private:
  TransformerNetConfig config_;

  // TensorRT 引擎和上下文 (Encoder 和 Decoder)
  std::shared_ptr<nvinfer1::ICudaEngine> encoder_engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> encoder_context_;
  std::shared_ptr<nvinfer1::ICudaEngine> decoder_engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> decoder_context_;

  // 图像尺寸相关
  int input_width_;
  int input_height_;
  int resized_width_;
  int resized_height_;
  float w_scale_;
  float h_scale_;

  // Transformer 特征尺寸
  int num_patches_;
  int patch_dim_;
  int sequence_length_;

  // Binding indices for encoder
  int image_input_index_;
  int encoder_output_index_;
  int positional_encoding_index_;
  
  // Binding indices for decoder
  int decoder_query_index_;
  int encoder_memory_index_;
  int keypoint_output_index_;
  int line_output_index_;
  int junction_output_index_;
  int descriptor_output_index_;

  // 中间结果存储
  std::vector<float> encoder_features_;
  std::vector<float> positional_encodings_;
  
  // 构建网络
  bool construct_encoder_network(
      TensorRTUniquePtr<nvinfer1::IBuilder> &builder, 
      TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
      TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config, 
      TensorRTUniquePtr<nvonnxparser::IParser> &parser) const;

  bool construct_decoder_network(
      TensorRTUniquePtr<nvinfer1::IBuilder> &builder, 
      TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
      TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config, 
      TensorRTUniquePtr<nvonnxparser::IParser> &parser) const;

  // 图像预处理
  bool preprocess_image(const tensorrt_buffer::BufferManager &buffers, 
                        const cv::Mat &image);
  
  // Patch 嵌入
  bool create_patch_embeddings(const cv::Mat &image, 
                               std::vector<float> &patch_embeddings);
  
  // 位置编码
  bool generate_positional_encoding(int sequence_length, 
                                    int d_model, 
                                    std::vector<float> &pos_encoding);

  // Encoder 推理
  bool run_encoder(const tensorrt_buffer::BufferManager &buffers);

  // Decoder 推理
  bool run_decoder(const tensorrt_buffer::BufferManager &buffers);

  // 输出后处理
  bool postprocess_output(const tensorrt_buffer::BufferManager &buffers, 
                          Eigen::Matrix<float, 259, Eigen::Dynamic> &features, 
                          std::vector<Eigen::Vector4d>& lines, 
                          Eigen::Matrix<float, 259, Eigen::Dynamic>& junctions, 
                          bool junction_detection);

  // 关键点解码
  bool decode_keypoints(const float* keypoint_logits, 
                        const float* descriptors, 
                        Eigen::Matrix<float, 259, Eigen::Dynamic> &features);

  // 线段解码
  bool decode_lines(const float* line_logits, 
                    std::vector<Eigen::Vector4d>& lines);

  // 交点解码
  bool decode_junctions(const float* junction_logits, 
                        const float* descriptors,
                        Eigen::Matrix<float, 259, Eigen::Dynamic> &junctions);

  // 非极大值抑制
  bool apply_nms(std::vector<Eigen::Vector4d>& lines, 
                 float iou_threshold = 0.5);

  // 特征提取和插值
  void extract_descriptors_at_points(const float *descriptor_map, 
                                     Eigen::Matrix<float, 259, Eigen::Dynamic> &features, 
                                     int h, int w, int descriptor_dim);

  // 工具函数
  std::vector<size_t> argsort(const std::vector<float> &scores, bool descending = true);
  int clip(int val, int max);
  float compute_line_iou(const Eigen::Vector4d& line1, const Eigen::Vector4d& line2);
  
  // Attention 可视化 (调试用)
  bool visualize_attention(const float* attention_weights, 
                          int num_heads, 
                          int seq_len,
                          const std::string& save_path);
};

typedef std::shared_ptr<TransformerNet> TransformerNetPtr;

#endif  // PLNET_TRANSFORMER_NET_H
