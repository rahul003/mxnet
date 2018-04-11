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
 * \file gradient_compression.h
 * \brief Gradient compression for kvstore
 * \author Rahul Huilgol
 */

#ifndef MXNET_KVSTORE_GRADIENT_COMPRESSION_H_
#define MXNET_KVSTORE_GRADIENT_COMPRESSION_H_
#include <dmlc/parameter.h>
#include <string>
#include <utility>
#include <vector>
#include "mxnet/ndarray.h"

namespace mxnet {
namespace kvstore {

enum class CompressionType {
  kNone, kTwoBit, kSignum, kMajority
};

enum class CompressionStep {
  kCpuAfterAggregation, kGpuAfterAggregation, kGpuBeforeAggregation
};

struct GradientCompressionParam : public dmlc::Parameter<GradientCompressionParam> {
  std::string type;
  std::string server_compression_type;
  std::string compression_step;
  float threshold;
  float beta;
  DMLC_DECLARE_PARAMETER(GradientCompressionParam) {
    DMLC_DECLARE_FIELD(type)
      .describe("Type of gradient compression to use, like `2bit` for example");
    DMLC_DECLARE_FIELD(threshold).set_default(0.5)
      .describe("Threshold to use for 2bit gradient compression");
    DMLC_DECLARE_FIELD(beta).set_default(0.9)
    .describe("Momentum parameter to use for efficient Signum compression");
    DMLC_DECLARE_FIELD(server_compression_type).set_default("none")
    .describe("Type of compression done by server after aggregating workers' gradients");
    DMLC_DECLARE_FIELD(compression_step).set_default("gpu_after_aggregation")
    .describe("When compression should be done for distributed case. Takes gpu_after_aggregation, "
              "gpu_before_aggregation, cpu_after_aggregation,");
  }
};

class GradientCompression {
 public:
  GradientCompression();

  virtual ~GradientCompression() {}

  /*!
   * \brief sets parameters for gradient compression
   * \param kwargs a vector of pair of strings. A pair represents key and value
   * of the parameter. Will be parsed by GradientCompressionParam
   */
  void SetParams(const std::vector<std::pair<std::string, std::string> >& kwargs);

  /*!
   * \brief returns type of compression on the worker
   */
  CompressionType get_type();

  /*!
   * \brief returns type of compression on the server
   */
  CompressionType get_server_compression_type();

  /*!
   * \brief returns when compression should be performed on the worker
   */
  CompressionStep get_compression_step();

  /*!
   * \brief returns as string the enum value of compression type
   */
  std::string get_type_str();

  /*!
   * \brief returns as string the enum value of server compression type
   */
  std::string get_server_compression_type_str();

  /*!
   * \brief returns as string the enum value of compression step
   */
  std::string get_compression_step_str();

  /*!
   * \brief sets when compression is to be done
   */
  inline void set_compression_step(CompressionStep s) { compression_step_ = s; }

  /*!
   * \brief sets two bit gradient compression
   * \param threshold float value used for thresholding gradients
   */
  void SetTwoBitCompression(const float threshold);

  /*!
   * \brief sets signum optimizer's compression
   * \param beta float value used for signum
   */
  void SetSignumCompression(const float beta);

  /*!
   * \brief encodes parameters of gc into a string
   */
  std::string EncodeParams();

  /*!
   * \brief decodes parameters of gc from a string and assigns them to member variables
   */
  void DecodeParams(const std::string &s);

  /*!
   * \brief returns compression factor, which is the factor by which size of gradient
   * reduces when using a particular type of compression
   */
  int GetCompressionFactor(const CompressionType& type);
  int GetCompressionFactor();

  /*!
   * \brief returns the size of compressed gradients given an original sized gradient array
   * using default compression type of worker
   */
  int64_t GetCompressedSize(const int64_t original_size);

  /*!
   * \brief returns the size of compressed gradients given an original sized gradient array
   * using given compression type
   */
  int64_t GetCompressedSize(const CompressionType& type, const int64_t original_size);

//  /*!
// * \brief returns the size of a block during compression by server.
// * A block is a unit of data which can not be split across servers during sharding.
// * For example, if using 3 bits for each gradient value, then we can not shard at 32bit boundaries for each float.
// * We would be sending incorrect and incomplete data if we did that.
// * \param num_workers
// */
//  int GetServerCompressionBlockSize(const int num_workers);
//
//  /*!
//   * \brief Gets the number of bits to use for compression by server
//   * \param num_workers number of workers from whom gradients are being accumulated
//   */
//  int GetServerCompressionNumBits(const int num_workers);
//
  /*!
   * \brief returns recompressed size after merging partially dequantized gradients from each worker
   * \param num_workers number of workers from whom gradients are being accumulated
   */
  int64_t GetServerRecompressedSize(const int64_t original_size);

  /*!
  * \brief Issues quantize operation to be scheduled by the engine
  * Compresses `from` into `to` and accumulates the quantization error
  * into 'residual', using the quantization of type `type_`
  * \param from the ndarray containing original data to be quantized
  * \param to the target ndarray which contains quantized data
  * \param residual the ndarray which accumulates quantization error
  * \param priority Priority of the action.
  * \param type of compression
  */
  void Quantize(const mxnet::NDArray &from, mxnet::NDArray *to,
                mxnet::NDArray *residual, const int priority, const CompressionType& type);

  /*!
   * \brief This method is used when the type for compression is the original type_
   */
  void Quantize(const mxnet::NDArray &from, mxnet::NDArray *to, mxnet::NDArray *residual, const int priority);

  /*!
   * \brief This method is used when type for compression is recompress_type_ . There is no residual for this
   */
  void Requantize(const mxnet::NDArray &from, mxnet::NDArray *to, const int priority);

  /*!
  * \brief Issues dequantize operation to be scheduled by the engine
  * Decompresses `from` into `to` using `type` and `threshold` passed
  * \param from the ndarray containing quantized data
  * \param to the target ndarray which contains final dequantized data
  * \param type of compression to use
  * \param threshold to be used in the case of 2bit compression
  * \param priority Priority of the action.
  */
  template <typename T>
  void Dequantize(const mxnet::NDArray &from, mxnet::NDArray *to, const int priority, const CompressionType type,
                  const T threshold);

  /*!
  * \brief Issues dequantize operation to be scheduled by the engine using the default compression type
   * for worker.
  * Decompresses `from` into `to` using worker's default parameters of `type` and `threshold`
  * \param from the ndarray containing quantized data
  * \param to the target ndarray which contains final dequantized data
  * \param priority Priority of the action.
  */
  void Dequantize(const mxnet::NDArray &from, mxnet::NDArray *to, const int priority);

  /*!
  * \brief Issues dequantize operation to be scheduled by the engine.
  * This is the final dequantization done by the worker when server compression type is set, or
  * by server when server_compression is not set.
  * \param from the ndarray containing quantized data
  * \param to the target ndarray which contains final dequantized data
  * \param type of compression to use
  * \param threshold to be used in the case of 2bit compression
  * \param priority Priority of the action.
  */
  void DequantizeFinal(const mxnet::NDArray &from, mxnet::NDArray *to, const int priority);

  void DequantizeForSum(const mxnet::NDArray &from, mxnet::NDArray *to, const int priority);
 private:
  /*!
   * \brief denotes the type of gradient compression which has been set
   */
  CompressionType type_;

  /*!
   * \brief denotes when to compress gradients on the worker
   */
  CompressionStep compression_step_;

  /*!
   * \brief denotes type of compression on the server.
   * In the case of single machine, this is not relevant.
   * In the case of multiple machines, this compression is done before pull
   */
  CompressionType server_compression_type_;

  /*!
   * \brief denotes threshold used for quantization and dequantization
   * Must be a positive value. All positive gradients will be thresholded to `threshold_` and
   * all negative gradients will be thresholded to -1*`threshold_`
   */
  float threshold_ = 0;

  /*!
  * \brief denotes the momentum parameter used for quantization and dequantization
  * Must be a number between  0 to 1.
  */
  float beta_ = 0;

};
}  // namespace kvstore
}  // namespace mxnet
#endif  // MXNET_KVSTORE_GRADIENT_COMPRESSION_H_
