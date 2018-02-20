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
 * \file gradient_compression.cu
 * \author Rahul Huilgol
 * \brief Implementation for gpu version of code
 */

#include "gradient_compression-inl.h"

namespace mxnet {
namespace kvstore {


void Quantize2BitImpl(mshadow::Stream<mshadow::gpu> *s,
                      const std::vector<mxnet::TBlob> &inputs,
                      const float threshold) {
  Quantize2BitKernelLaunch(s, inputs, threshold);
}

void QuantizeSignumImpl(mshadow::Stream<mshadow::gpu> *s,
                        const std::vector<mxnet::TBlob> &inputs,
                        const float beta) {
  QuantizeSignumKernelLaunch(s, inputs, beta);
}

void QuantizeLogKImpl(mshadow::Stream<mshadow::gpu> *s,
                      const std::vector<mxnet::TBlob> &inputs,
                      const int num_workers) {
  QuantizeLogKKernelLaunch(s, inputs, num_workers, inputs[0].Size());
}

template <typename DType>
void Dequantize2BitImpl(mshadow::Stream<mshadow::gpu> *s,
                        const std::vector<mxnet::TBlob> &inputs,
                        const DType threshold) {
  Dequantize2BitKernelLaunch(s, inputs, threshold);
}

void DequantizeSignumImpl(mshadow::Stream<mshadow::gpu> *s,
                          const std::vector<mxnet::TBlob> &inputs) {
  if (inputs[1].type_flag_ == mshadow::kFloat32) {
    DequantizeSignumKernelLaunch(s, inputs, (float) 1.0);
  } else if (inputs[1].type_flag_ == mshadow::kInt32) {
    DequantizeSignumKernelLaunch(s, inputs, (int) 1);
  } else {
    LOG(FATAL) << "Unhandled type";
  }
}

void DequantizeLogKImpl(mshadow::Stream<mshadow::gpu> *s,
                        const std::vector<mxnet::TBlob> &inputs,
                        const float threshold,
                        const int num_workers) {
  DequantizeLogKKernelLaunch(s, inputs, threshold, num_workers, inputs[1].Size());
}

}  // namespace kvstore
}  // namespace mxnet
