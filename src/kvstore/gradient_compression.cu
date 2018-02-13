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
void Quantize2BitImpl(mshadow::Stream<gpu>* s, const std::vector<TBlob>& inputs,
                      const float threshold) {
  Quantize2BitKernelLaunch(s, inputs, threshold);
}

void Dequantize2BitImpl(mshadow::Stream<gpu>* s, const std::vector<TBlob>& inputs,
                        const float threshold) {
  Dequantize2BitKernelLaunch(s, inputs, threshold);
}

void Dequantize2BitForSumImpl(mshadow::Stream<mshadow::gpu> *s,
                                     const std::vector<mxnet::TBlob> &inputs) {
  Dequantize2BitForSumKernelLaunch(s, inputs);
}

void RequantizeImpl(mshadow::Stream<mshadow::gpu> *s,
                           const std::vector<mxnet::TBlob> &inputs,
                           const int num_workers,
                           const int original_size) {
  RequantizeKernelLaunch(s, inputs, num_workers, original_size);
}

void DerequantizeImpl(mshadow::Stream<mshadow::gpu> *s,
                             const std::vector<mxnet::TBlob> &inputs,
                             const float threshold,
                             const int num_workers,
                             const int original_size) {
  DerequantizeKernelLaunch(s, inputs, threshold, num_workers, original_size);
}


void QuantizeSignumImpl(mshadow::Stream<gpu>* s, const std::vector<TBlob>& inputs,
                      const float beta) {
  QuantizeSignumKernelLaunch(s, inputs, beta);
}

void DequantizeSignumImpl(mshadow::Stream<gpu>* s, const std::vector<TBlob>& inputs) {
  DequantizeSignumKernelLaunch(s, inputs);
}

}  // namespace kvstore
}  // namespace mxnet
