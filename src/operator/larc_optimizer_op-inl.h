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
 *  Copyright (c) 2018 by Contributors
 * \file larc_optimizer-inl.h
 * \brief LARC Optimizer operators
 * \author Clement Fuji Tsang
 */

#ifndef MXNET_OPERATOR_LARC_OPTIMIZER_OP_INL_H_
#define MXNET_OPERATOR_LARC_OPTIMIZER_OP_INL_H_

#include <algorithm>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <mshadow/base.h>
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <vector>
#include "../common/cuda_utils.h"
#include "./operator_common.h"
#include "./mshadow_op.h"
#include "./elemwise_op_common.h"
#include "mxnet_op.h"
#include "./tensor/init_op.h"
#include "./tensor/util/tensor_util-inl.h"
#include "./optimizer_op-inl.h"

namespace mxnet {
namespace op {

#ifdef __CUDACC__

using mshadow::cuda::kMaxThreadsPerBlock;

template<typename Dtype, typename Mtype>
__global__ void gpu_sum_sq_kernel(const int N, const Dtype* __restrict__ x, Mtype* out) {
  __shared__ Mtype cache[kMaxThreadsPerBlock];
  const int tidx = threadIdx.x;
  cache[tidx] = 0.;
  // __syncthreads();
  for (int i = tidx; i < N; i += blockDim.x) {
    cache[tidx] += static_cast<Mtype>(x[i]) * static_cast<Mtype>(x[i]);
  }
  __syncthreads();
  for (int s = mshadow::cuda::kMaxThreadsPerBlock / 2; s > 0; s >>= 1) {
    if (tidx < s) cache[tidx] += cache[tidx + s];
    __syncthreads();
  }
  if (tidx == 0) *out = cache[tidx];
}

template<typename Dtype, typename Mtype>
inline void sum_sq(mshadow::Stream<gpu> *s, const int& n,
                   const Dtype* X, Mtype* out) {
  LOG(FATAL) << "Not implemented!";
}

template<>
inline void sum_sq(mshadow::Stream<gpu> *s, const int& n,
                   const float* X, float* out) {
  cublasSetPointerMode(mxnet_op::Stream<gpu>::GetBlasHandle(s),
                       CUBLAS_POINTER_MODE_HOST);
  cublasStatus_t err = cublasSdot(mxnet_op::Stream<gpu>::GetBlasHandle(s),
                                  n, X, 1, X, 1, out);
  CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Dot fail";
}

template<>
inline void sum_sq(mshadow::Stream<gpu> *s, const int& n,
                   const double* X, double* out) {
  cublasSetPointerMode(mxnet_op::Stream<gpu>::GetBlasHandle(s),
                       CUBLAS_POINTER_MODE_HOST);
  cublasStatus_t err = cublasDdot(mxnet_op::Stream<gpu>::GetBlasHandle(s),
                                  n, X, 1, X, 1, out);
  CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Dot fail";
}

template<>
inline void sum_sq(mshadow::Stream<gpu> *s, const int& n,
                   const mshadow::half::half_t* X, mshadow::half::half_t* out) {
  cublasSetPointerMode(mxnet_op::Stream<gpu>::GetBlasHandle(s),
                       CUBLAS_POINTER_MODE_HOST);
  cublasStatus_t err = cublasDotEx(mxnet_op::Stream<gpu>::GetBlasHandle(s), n,
                                   reinterpret_cast<const __half*>(X), CUDA_R_16F, 1,
                                   reinterpret_cast<const __half*>(X), CUDA_R_16F, 1,
                                   reinterpret_cast<__half*>(out), CUDA_R_16F, CUDA_R_32F);
  CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Dot fail";
}

template<>
inline void sum_sq(mshadow::Stream<gpu> *s, const int& n,
                   const mshadow::half::half_t* X, float* out) {
  gpu_sum_sq_kernel<<<1, kMaxThreadsPerBlock, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
      n, reinterpret_cast<const __half*>(X), out);
  MSHADOW_CUDA_POST_KERNEL_CHECK(gpu_sum_sq_kernel);
}

#endif  // __CUDACC__

struct LARCParam : public dmlc::Parameter<LARCParam> {
  float lr;
  float wd;
  float rescale_grad;
  float clip_gradient;
  float trust_coeff;
  DMLC_DECLARE_PARAMETER(LARCParam) {
    DMLC_DECLARE_FIELD(lr)
    .describe("Learning rate");
    DMLC_DECLARE_FIELD(wd)
    .set_default(0.0f)
    .describe("Weight decay augments the objective function with a "
              "regularization term that penalizes large weights. "
              "The penalty scales with the square of the magnitude of each weight.");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("Rescale gradient to grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(trust_coeff)
    .set_default(0.01f)
    .describe("Trust coefficient");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("Clip gradient to the range of [-clip_gradient, clip_gradient] "
              "If clip_gradient <= 0, gradient clipping is turned off. "
              "grad = max(min(grad, clip_gradient), -clip_gradient).");
  }
};

struct LARCMomParam : public dmlc::Parameter<LARCMomParam> {
  float lr;
  float momentum;
  float wd;
  float rescale_grad;
  float clip_gradient;
  float trust_coeff;
  DMLC_DECLARE_PARAMETER(LARCMomParam) {
    DMLC_DECLARE_FIELD(lr)
    .describe("Learning rate");
    DMLC_DECLARE_FIELD(momentum)
    .set_default(0.0f)
    .describe("The decay rate of momentum estimates at each epoch.");
    DMLC_DECLARE_FIELD(wd)
    .set_default(0.0f)
    .describe("Weight decay augments the objective function with a "
              "regularization term that penalizes large weights. "
              "The penalty scales with the square of the magnitude of each weight.");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("Rescale gradient to grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(trust_coeff)
    .set_default(0.01f)
    .describe("Trust coefficient");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("Clip gradient to the range of [-clip_gradient, clip_gradient] "
              "If clip_gradient <= 0, gradient clipping is turned off. "
              "grad = max(min(grad, clip_gradient), -clip_gradient).");
  }
};


template<void (*T)(const nnvm::NodeAttrs&,
                   const OpContext&,
                   const std::vector<TBlob>&,
                   const std::vector<OpReqType>&,
                   const std::vector<TBlob>&)>
inline void LARCUpdate(const nnvm::NodeAttrs& attrs,
                       const OpContext &ctx,
                       const std::vector<TBlob> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &outputs) {
  LOG(FATAL) << "Not implemented!";
}

template<>
inline void LARCUpdate<SGDUpdate<cpu> >(const nnvm::NodeAttrs& attrs,
                                        const OpContext &ctx,
                                        const std::vector<TBlob> &inputs,
                                        const std::vector<OpReqType> &req,
                                        const std::vector<TBlob> &outputs) {
  LOG(FATAL) << "not implemented!";
}


template<>
inline void LARCUpdate<SGDMomUpdate<cpu> >(const nnvm::NodeAttrs& attrs,
                                           const OpContext &ctx,
                                           const std::vector<TBlob> &inputs,
                                           const std::vector<OpReqType> &req,
                                           const std::vector<TBlob> &outputs) {
  LOG(FATAL) << "not implemented!";
}

template<>
inline void LARCUpdate<MP_SGDUpdate<cpu> >(const nnvm::NodeAttrs& attrs,
                                           const OpContext &ctx,
                                           const std::vector<TBlob> &inputs,
                                           const std::vector<OpReqType> &req,
                                           const std::vector<TBlob> &outputs) {
  LOG(FATAL) << "not implemented!";
}

template<>
inline void LARCUpdate<MP_SGDMomUpdate<cpu> >(const nnvm::NodeAttrs& attrs,
                                              const OpContext &ctx,
                                              const std::vector<TBlob> &inputs,
                                              const std::vector<OpReqType> &req,
                                              const std::vector<TBlob> &outputs) {
  LOG(FATAL) << "not implemented!";
}

#ifdef __CUDACC__

template<>
inline void LARCUpdate<SGDUpdate<gpu> >(const nnvm::NodeAttrs& attrs,
                                        const OpContext &ctx,
                                        const std::vector<TBlob> &inputs,
                                        const std::vector<OpReqType> &req,
                                        const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  LARCParam param = nnvm::get<LARCParam>(attrs.parsed);
  Stream<gpu>* s = ctx.get_stream<gpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<gpu, 2, DType> weight = inputs[0].FlatTo2D<gpu, DType>(s);
    Tensor<gpu, 2, DType> grad = inputs[1].FlatTo2D<gpu, DType>(s);
    Tensor<gpu, 2, DType> out = outputs[0].FlatTo2D<gpu, DType>(s);
    DType sq_weight, sq_grad, final_lr;
    sum_sq(s, weight.shape_.Size(), weight.dptr_, &sq_weight);
    sum_sq(s, grad.shape_.Size(), grad.dptr_, &sq_grad);
    if (sq_weight > 0.f && sq_grad > 0.) {
      final_lr = static_cast<DType>(param.trust_coeff) *
          std::sqrt(sq_weight / sq_grad);
    } else {
      final_lr = DType(1.);
    }
    final_lr = std::min(final_lr, static_cast<DType>(param.lr));
    Kernel<SGDKernel, gpu>::Launch(s, weight.shape_.Size(), out.dptr_, weight.dptr_,
      grad.dptr_, static_cast<DType>(param.clip_gradient),
      final_lr, static_cast<DType>(param.wd),
      static_cast<DType>(param.rescale_grad), req[0]);
    });
}

template<>
inline void LARCUpdate<SGDMomUpdate<gpu> >(const nnvm::NodeAttrs& attrs,
                                           const OpContext &ctx,
                                           const std::vector<TBlob> &inputs,
                                           const std::vector<OpReqType> &req,
                                           const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  LARCMomParam param = nnvm::get<LARCMomParam>(attrs.parsed);
  Stream<gpu>* s = ctx.get_stream<gpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<gpu, 2, DType> weight = inputs[0].FlatTo2D<gpu, DType>(s);
    Tensor<gpu, 2, DType> grad = inputs[1].FlatTo2D<gpu, DType>(s);
    Tensor<gpu, 2, DType> mom = inputs[2].FlatTo2D<gpu, DType>(s);
    Tensor<gpu, 2, DType> out = outputs[0].FlatTo2D<gpu, DType>(s);
    DType sq_weight, sq_grad, final_lr;
    sum_sq(s, weight.shape_.Size(), weight.dptr_, &sq_weight);
    sum_sq(s, grad.shape_.Size(), grad.dptr_, &sq_grad);
    if (sq_weight > 0.f && sq_grad > 0.) {
      final_lr = static_cast<DType>(param.trust_coeff) *
          std::sqrt(sq_weight / sq_grad);
    } else {
      final_lr = DType(1.);
    }
    final_lr = std::min(final_lr, static_cast<DType>(param.lr));
    Kernel<SGDMomKernel, gpu>::Launch(s, weight.shape_.Size(), out.dptr_, mom.dptr_, weight.dptr_,
      grad.dptr_, static_cast<DType>(param.clip_gradient), static_cast<DType>(param.momentum),
      final_lr, static_cast<DType>(param.wd), static_cast<DType>(param.rescale_grad),
      req[0]);
    });
}

template<>
inline void LARCUpdate<MP_SGDUpdate<gpu> >(const nnvm::NodeAttrs& attrs,
                                           const OpContext &ctx,
                                           const std::vector<TBlob> &inputs,
                                           const std::vector<OpReqType> &req,
                                           const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  LARCParam param = nnvm::get<LARCParam>(attrs.parsed);
  Stream<gpu>* s = ctx.get_stream<gpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<gpu, 2, DType> weight = inputs[0].FlatTo2D<gpu, DType>(s);
    Tensor<gpu, 2, DType> grad = inputs[1].FlatTo2D<gpu, DType>(s);
    Tensor<gpu, 2, float> weight32 = inputs[2].FlatTo2D<gpu, float>(s);
    Tensor<gpu, 2, DType> out = outputs[0].FlatTo2D<gpu, DType>(s);
    float sum_sq_data[2], final_lr;
    Tensor<gpu, 1, float> gpu_temp =
        ctx.requested[0].get_space_typed<gpu, 1, float>(Shape1(2), s);
    sum_sq(s, weight.shape_.Size(), weight.dptr_, gpu_temp.dptr_);
    sum_sq(s, grad.shape_.Size(), grad.dptr_, &(gpu_temp.dptr_[1]));
    MSHADOW_CUDA_CALL(cudaMemcpyAsync(sum_sq_data, gpu_temp.dptr_, sizeof(float) * 2,
                                      cudaMemcpyDeviceToHost,
                                      mshadow::Stream<gpu>::GetStream(s)));
    MSHADOW_CUDA_CALL(cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(s)));
    float sum_sq_weight = sum_sq_data[0], sum_sq_grad = sum_sq_data[1]; 
    if (sum_sq_weight > 0.f && sum_sq_grad > 0.f) {
      final_lr = param.trust_coeff * std::sqrt(sum_sq_weight / sum_sq_grad);
    } else {
      final_lr = 1.f;
    }
    final_lr = std::min(final_lr, param.lr);
    Kernel<MP_SGDKernel, gpu>::Launch(s, weight.shape_.Size(), out.dptr_,
      weight.dptr_, grad.dptr_, weight32.dptr_, param.clip_gradient,
      final_lr, param.wd, param.rescale_grad, req[0]);
  });
}

template<>
inline void LARCUpdate<MP_SGDMomUpdate<gpu> >(const nnvm::NodeAttrs& attrs,
                                              const OpContext &ctx,
                                              const std::vector<TBlob> &inputs,
                                              const std::vector<OpReqType> &req,
                                              const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  LARCMomParam param = nnvm::get<LARCMomParam>(attrs.parsed);
  Stream<gpu>* s = ctx.get_stream<gpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<gpu, 2, DType> weight = inputs[0].FlatTo2D<gpu, DType>(s);
    Tensor<gpu, 2, DType> grad = inputs[1].FlatTo2D<gpu, DType>(s);
    Tensor<gpu, 2, float> mom = inputs[2].FlatTo2D<gpu, float>(s);
    Tensor<gpu, 2, float> weight32 = inputs[3].FlatTo2D<gpu, float>(s);
    Tensor<gpu, 2, DType> out = outputs[0].FlatTo2D<gpu, DType>(s);
    float sum_sq_data[2], final_lr;
    Tensor<gpu, 1, float> gpu_temp =
        ctx.requested[0].get_space_typed<gpu, 1, float>(Shape1(2), s);
    sum_sq(s, weight.shape_.Size(), weight.dptr_, gpu_temp.dptr_);
    sum_sq(s, grad.shape_.Size(), grad.dptr_, &(gpu_temp.dptr_[1]));
    MSHADOW_CUDA_CALL(cudaMemcpyAsync(sum_sq_data, gpu_temp.dptr_, sizeof(float) * 2,
                                 cudaMemcpyDeviceToHost,
                                 mshadow::Stream<gpu>::GetStream(s)));
    MSHADOW_CUDA_CALL(cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(s)));
    float sum_sq_weight = sum_sq_data[0], sum_sq_grad = sum_sq_data[1];
    if (sum_sq_weight > 0.f && sum_sq_grad > 0.f) {
      final_lr = param.trust_coeff *
                  std::sqrt(sum_sq_weight / sum_sq_grad);
    } else {
      final_lr = 1.f;
    }
    final_lr = std::min(final_lr, param.lr);
    Kernel<MP_SGDMomKernel, gpu>::Launch(s, weight.shape_.Size(), out.dptr_, mom.dptr_,
      weight.dptr_, grad.dptr_, weight32.dptr_, param.clip_gradient, param.momentum,
      final_lr, param.wd, param.rescale_grad, req[0]);
  });
}

#endif  // __CUDACC__

}  // namespace op
}  // namespace mxnet


#endif  // MXNET_OPERATOR_LARC_OPTIMIZER_OP_INL_H_
