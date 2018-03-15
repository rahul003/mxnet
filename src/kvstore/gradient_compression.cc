/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file gradient_compression.cc
 * \brief Gradient compression for kvstore
 * \author Rahul Huilgol
 */

#include <sstream>
#include <vector>
#include "gradient_compression.h"
#include "gradient_compression-inl.h"
#include "../../include/mxnet/ndarray.h"

namespace mxnet {
namespace kvstore {

/*!
 * \brief Splits a string into smaller strings using char as delimiter
 * Example: "a,b,c,,d" is split into ["a","b","c","","d"]
 * \param s string to split
 * \param delim char to split string around
 * \param result container for tokens extracted after splitting
 */
template<typename Out>
void split(const std::string &s, const char delim, Out result) {
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    *(result++) = item;
  }
}

DMLC_REGISTER_PARAMETER(GradientCompressionParam);

GradientCompression::GradientCompression() {
  type_ = CompressionType::kNone;
  server_compression_type_ = CompressionType::kNone;
}

void GradientCompression::SetParams(const std::vector<std::pair<std::string, std::string> >
                                    & kwargs) {
  GradientCompressionParam params;
  params.InitAllowUnknown(kwargs);
  CHECK_GT(params.threshold, 0) << "threshold must be greater than 0";
  if (params.type == "2bit") {
    SetTwoBitCompression(params.threshold);
  } else if (params.type == "signum") {
    SetSignumCompression(params.beta);
  } else {
    LOG(FATAL) << "Unknown type for gradient compression " << params.type;
  }

  if (params.server_compression_type == "majority") {
    server_compression_type_ = CompressionType::kMajority;
  } else if (params.server_compression_type == "none") {
    server_compression_type_ = CompressionType::kNone;
  } else {
    LOG(FATAL) << "Unknown type for recompress type in gradient compression " << params.server_compression_type;
  }

  if (params.compression_step == "gpu_after_aggregation") {
    compression_step_ = CompressionStep::kGpuAfterAggregation;
  } else if (params.compression_step == "cpu_after_aggregation") {
    compression_step_ = CompressionStep::kCpuAfterAggregation;
  } else if (params.compression_step == "gpu_before_aggregation") {
    compression_step_ = CompressionStep::kGpuBeforeAggregation;
  } else {
    LOG(FATAL) << "Unknown compression step " << params.compression_step;
  }
}

CompressionType GradientCompression::get_type() {
  return type_;
}

CompressionType GradientCompression::get_server_compression_type() {
  return server_compression_type_;
}

CompressionStep GradientCompression::get_compression_step() {
  return compression_step_;
}

std::string GradientCompression::get_type_str() {
  return std::to_string(static_cast<int>(type_));
}

std::string GradientCompression::get_server_compression_type_str() {
  return std::to_string(static_cast<int>(server_compression_type_));
}

std::string GradientCompression::get_compression_step_str() {
  return std::to_string(static_cast<int>(compression_step_));
}

void GradientCompression::SetTwoBitCompression(const float threshold) {
  type_ = CompressionType::kTwoBit;
  threshold_ = threshold;
}

void GradientCompression::SetSignumCompression(const float beta) {
  type_ = CompressionType::kSignum;
  beta_ = beta;
}

void GradientCompression::SetNumWorkers(const int num_workers) {
  num_workers_ = num_workers;
}

// Only encodes params that server needs
std::string GradientCompression::EncodeParams() {
  using namespace std;  // to reduce length of next line
  string rval = get_type_str();
  if (type_ == CompressionType::kTwoBit) {
    rval += "," + to_string(threshold_);
  } else if (type_ == CompressionType::kSignum) {
    rval += "," + to_string(beta_);
  }
  rval +=  "," + get_server_compression_type_str();
  return rval;
}

void GradientCompression::DecodeParams(const std::string &s) {
  std::vector<std::string> elems;
  split(s, ',', std::back_inserter(elems));
  type_ = static_cast<CompressionType>(stoi(elems[0]));
  CHECK_GE(elems.size(), 2);
  if (type_ == CompressionType::kTwoBit) {
    threshold_ = stof(elems[1]);
  } else if(type_ == CompressionType::kSignum) {
    beta_ = stof(elems[1]);
  }
  if (elems.size() > 2) {
    server_compression_type_ = static_cast<CompressionType>(stoi(elems[2]));
  }
}

int GradientCompression::GetCompressionFactor(const CompressionType& type) {
  if (type == CompressionType::kTwoBit) {
    return 16;
  } else if (type == CompressionType::kSignum || type == CompressionType::kMajority) {
    return 32;
  } else {
    LOG(FATAL) << "Unknown compression type: " << static_cast<int>(type);
    return 0;
  }
}

int64_t GradientCompression::GetCompressedSize(const int64_t original_size) {
  const int bits = GetCompressionFactor(type_);
  return ((original_size % bits == 0) ?
          original_size / bits :
          original_size / bits + 1);
}

int64_t GradientCompression::GetCompressedSize(const CompressionType& type, const int64_t original_size) {
  const int bits = GetCompressionFactor(type);
  return ((original_size % bits == 0) ?
          original_size / bits :
          original_size / bits + 1);
}

int64_t GradientCompression::GetServerRecompressedSize(const int64_t original_size) {
  if (server_compression_type_ == CompressionType::kMajority) {
    return GetCompressedSize(server_compression_type_, original_size);
  } else if (server_compression_type_ == CompressionType::kNone) {
    return original_size;
  } else {
    LOG(FATAL) << "Unsupported compression type: " << get_server_compression_type_str();
  }
  return 0;
}

// returns number of floats in requantized data Block for logk
int GradientCompression::GetServerCompressionBlockSize(const int num_workers) {
  return 1;
}

int GradientCompression::GetServerCompressionNumBits(const int num_workers) {
  if (server_compression_type_ == CompressionType::kMajority) {
    return 1;
  } else if (server_compression_type_ == CompressionType::kNone) {
    return 32;
  } else {
    LOG(FATAL) << "Check server compression type " << get_server_compression_type_str();
  }
}

void GradientCompression::Quantize(const mxnet::NDArray &from,
                                   mxnet::NDArray *to,
                                   mxnet::NDArray *residual,
                                   const int priority) {
  Quantize(from, to, residual, priority, type_);
}

void GradientCompression::Requantize(const mxnet::NDArray &from,
                                   mxnet::NDArray *to,
                                   const int priority) {
  Quantize(from, to, nullptr, priority, server_compression_type_);
}

void GradientCompression::Quantize(const mxnet::NDArray &from,
                                   mxnet::NDArray *to,
                                   mxnet::NDArray *residual,
                                   const int priority,
                                   const CompressionType type) {
  CHECK(from.shape().ndim() != 0) << "source operand has zero dimension shape";
  CHECK(to->shape().ndim() != 0) << "destination operand has zero dimension shape";
  const int a = from.ctx().dev_mask();
  const int b = to->ctx().dev_mask();
  if (type == CompressionType::kTwoBit) {
    CHECK(residual->shape().ndim() != 0) << "residual operand has zero dimension shape";
    const float threshold = threshold_;
    if (a == mshadow::cpu::kDevMask && b == mshadow::cpu::kDevMask) {
      mxnet::Engine::Get()->PushSync([from, to, residual, threshold](mxnet::RunContext ctx) {
        std::vector<mxnet::TBlob> inputs = {from.data(), residual->data(), to->data()};
        Quantize2BitImpl(ctx.get_stream<mshadow::cpu>(), inputs, threshold);
      }, from.ctx(), {from.var()}, {to->var(), residual->var()},
      mxnet::FnProperty::kNormal, priority, PROFILER_MESSAGE("QuantizeCPU"));
    } else {
#if MXNET_USE_CUDA
      if (a == mshadow::gpu::kDevMask && b == mshadow::gpu::kDevMask) {
        mxnet::Engine::Get()->PushSync([from, to, residual, threshold](mxnet::RunContext ctx) {
          std::vector<mxnet::TBlob> inputs = {from.data(), residual->data(), to->data()};
          Quantize2BitImpl(ctx.get_stream<mshadow::gpu>(), inputs, threshold);
          // Wait GPU kernel to complete
          ctx.get_stream<mshadow::gpu>()->Wait();
        }, from.ctx(), {from.var()}, {to->var(), residual->var()},
        mxnet::FnProperty::kNormal, priority, PROFILER_MESSAGE("QuantizeGPU"));
      } else {
        LOG(FATAL) << "unknown device mask";
      }
#else
    LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
    }
  } else if (type == CompressionType::kSignum) {
    CHECK(residual->shape().ndim() != 0) << "residual operand has zero dimension shape";
    const float beta = beta_;
    if (a == mshadow::cpu::kDevMask && b == mshadow::cpu::kDevMask) {
      mxnet::Engine::Get()->PushSync([from, to, residual, beta](mxnet::RunContext ctx) {
        std::vector<mxnet::TBlob> inputs = {from.data(), residual->data(), to->data()};
        QuantizeSignumImpl(ctx.get_stream<mshadow::cpu>(), inputs, beta);
      }, from.ctx(), {from.var()}, {to->var(), residual->var()},
      mxnet::FnProperty::kNormal, priority, PROFILER_MESSAGE("QuantizeCPU"));
    } else {
#if MXNET_USE_CUDA
      if (a == mshadow::gpu::kDevMask && b == mshadow::gpu::kDevMask) {
        mxnet::Engine::Get()->PushSync([from, to, residual, beta](mxnet::RunContext ctx) {
          std::vector<mxnet::TBlob> inputs = {from.data(), residual->data(), to->data()};
          QuantizeSignumImpl(ctx.get_stream<mshadow::gpu>(), inputs, beta);
          // Wait GPU kernel to complete
          ctx.get_stream<mshadow::gpu>()->Wait();
        }, from.ctx(), {from.var()}, {to->var(), residual->var()},
        mxnet::FnProperty::kNormal, priority, PROFILER_MESSAGE("QuantizeGPU"));
      } else {
        LOG(FATAL) << "unknown device mask";
      }
#else
    LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
    }
  } else if (type == CompressionType::kMajority) {
    if (a == mshadow::cpu::kDevMask && b == mshadow::cpu::kDevMask) {
      mxnet::Engine::Get()->PushSync([from, to, type](mxnet::RunContext ctx) {
         std::vector<mxnet::TBlob> inputs = {from.data(), to->data()};
         QuantizeFromIntSumImpl(ctx.get_stream<mshadow::cpu>(), inputs, type);
       }, from.ctx(), {from.var()}, {to->var()},
       mxnet::FnProperty::kNormal, priority, PROFILER_MESSAGE("QuantizeFromIntSumCPU"));
    } else {
#if MXNET_USE_CUDA
      if (a == mshadow::gpu::kDevMask && b == mshadow::gpu::kDevMask) {
        mxnet::Engine::Get()->PushSync([from, to, type](mxnet::RunContext ctx) {
          std::vector<mxnet::TBlob> inputs = {from.data(), to->data()};
          QuantizeFromIntSumImpl(ctx.get_stream<mshadow::gpu>(), inputs, type);
          // Wait GPU kernel to complete
          ctx.get_stream<mshadow::gpu>()->Wait();
        }, from.ctx(), {from.var()}, {to->var()},
        mxnet::FnProperty::kNormal, priority, PROFILER_MESSAGE("QuantizeFromIntSumGPU"));
      } else {
        LOG(FATAL) << "unknown device mask";
      }
#else
      LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
    }
  } else {
    LOG(FATAL) << "Unsupported quantization of type " << get_type_str();
  }
}


void GradientCompression::Dequantize(const mxnet::NDArray &from, mxnet::NDArray *to,
                                     const int priority) {
  Dequantize(from, to, priority, type_, threshold_);
}

void GradientCompression::DequantizeFinal(const mxnet::NDArray &from, mxnet::NDArray *to,
                                     const int priority) {
  *to = 0;
  Dequantize(from, to, priority, server_compression_type_, threshold_);
}

template<typename T>
void GradientCompression::Dequantize(const mxnet::NDArray &from,
                                     mxnet::NDArray *to,
                                     const int priority,
                                     const CompressionType type,
                                     const T threshold) {
  CHECK(from.shape().ndim() != 0) << "source operands has zero dimension shape";
  CHECK(to->shape().ndim() != 0) << "destination operand has zero dimension shape";
  const int a = from.ctx().dev_mask();
  const int b = to->ctx().dev_mask();
  if (type == CompressionType::kTwoBit) {
    if (a == mshadow::cpu::kDevMask && b == mshadow::cpu::kDevMask) {
      mxnet::Engine::Get()->PushSync([from, to, threshold](mxnet::RunContext ctx) {
        std::vector<mxnet::TBlob> inputs = {from.data(), to->data()};
        Dequantize2BitImpl(ctx.get_stream<mshadow::cpu>(), inputs, &threshold);
      }, from.ctx(), {from.var()}, {to->var()},
      mxnet::FnProperty::kNormal, priority, PROFILER_MESSAGE("DequantizeCPU"));
    } else {
#if MXNET_USE_CUDA
      if (a == mshadow::gpu::kDevMask && b == mshadow::gpu::kDevMask) {
        mxnet::Engine::Get()->PushSync([from, to, threshold](mxnet::RunContext ctx) {
          std::vector<mxnet::TBlob> inputs = {from.data(), to->data()};
          Dequantize2BitImpl(ctx.get_stream<mshadow::gpu>(), inputs, &threshold);
          // Wait GPU kernel to complete
          ctx.get_stream<mshadow::gpu>()->Wait();
        }, from.ctx(), {from.var()}, {to->var()},
        mxnet::FnProperty::kNormal, priority, PROFILER_MESSAGE("DequantizeGPU"));
      } else {
        LOG(FATAL) << "unknown device mask";
      }
#else
      LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
    }
  } else if (type == CompressionType::kSignum || type == CompressionType::kMajority) {
    if (a == mshadow::cpu::kDevMask && b == mshadow::cpu::kDevMask) {
      mxnet::Engine::Get()->PushSync([from, to, threshold](mxnet::RunContext ctx) {
        std::vector<mxnet::TBlob> inputs = {from.data(), to->data()};
        DequantizeSignumImpl(ctx.get_stream<mshadow::cpu>(), inputs);
      }, from.ctx(), {from.var()}, {to->var()},
      mxnet::FnProperty::kNormal, priority, PROFILER_MESSAGE("DequantizeCPU"));
    } else {
#if MXNET_USE_CUDA
      if (a == mshadow::gpu::kDevMask && b == mshadow::gpu::kDevMask) {
        mxnet::Engine::Get()->PushSync([from, to, threshold](mxnet::RunContext ctx) {
          std::vector<mxnet::TBlob> inputs = {from.data(), to->data()};
          DequantizeSignumImpl(ctx.get_stream<mshadow::gpu>(), inputs);
          // Wait GPU kernel to complete
          ctx.get_stream<mshadow::gpu>()->Wait();
        }, from.ctx(), {from.var()}, {to->var()},
        mxnet::FnProperty::kNormal, priority, PROFILER_MESSAGE("DequantizeGPU"));
      } else {
        LOG(FATAL) << "unknown device mask";
      }
#else
      LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
    }
  } else {
    LOG(FATAL) << "Unsupported dequantization of type " << get_type_str();
  }
}

void GradientCompression::DequantizeForSum(const mxnet::NDArray &from, mxnet::NDArray *to,
                                           const int priority) {
  Dequantize(from, to , priority, type_, (int) 1);
}


}  // namespace kvstore
}  // namespace mxnet

