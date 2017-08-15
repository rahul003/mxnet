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
 * \file two_bit_quantize_sim-inl.h
 * \brief implementation of quantize_2bit operation (simulation)
 */
#ifndef MXNET_OPERATOR_CONTRIB_TWO_BIT_QUANTIZE_SIM_INL_H_
#define MXNET_OPERATOR_CONTRIB_TWO_BIT_QUANTIZE_SIM_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <limits>
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

struct quantize_2bit {
  template<typename DstDType, typename SrcDType>
  MSHADOW_XINLINE static void Map(int i,
                                  DstDType *out,
                                  float *oneg_shreshold,
                                  float *opos_shreshold,
                                  const SrcDType *in,
                                  const float *ineg_shreshold,
                                  const float *ipos_shreshold) {
    out[i] = in[i] >= *ipos_shreshold ? *ipos_shreshold :
            (in[i] <= *ineg_shreshold ? *ineg_shreshold : 0.0);
    *oneg_shreshold = *ineg_shreshold;
    *opos_shreshold = *ipos_shreshold;
  }
};

template<typename xpu>
void Quantize2BitCompute(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();

  typedef float DstDType;
  typedef float SrcDType;
  Kernel<quantize_2bit, xpu>::Launch(s, outputs[0].Size(),
    outputs[0].dptr<DstDType>(),
    outputs[1].dptr<float>(),
    outputs[2].dptr<float>(),
    inputs[0].dptr<SrcDType>(),
    inputs[1].dptr<float>(),
    inputs[2].dptr<float>());
}

inline bool Quantize2BitShape(const nnvm::NodeAttrs& attrs,
                              std::vector<TShape> *in_attrs,
                              std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);
  CHECK(!shape_is_none(in_attrs->at(0)));
  for (size_t i = 1; i < 3; ++i) {
    CHECK(shape_is_scalar(in_attrs->at(i)));
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, TShape{1});
  SHAPE_ASSIGN_CHECK(*out_attrs, 2, TShape{1});
  return true;
}

inline bool Quantize2BitType(const nnvm::NodeAttrs& attrs,
                             std::vector<int> *in_attrs,
                             std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);
  CHECK_EQ((*in_attrs)[0], mshadow::kFloat32)
    << "`quantize` only supports float32 input for now";
  CHECK_EQ((*in_attrs)[1], mshadow::kFloat32)
    << "the second input of `quantize` should be a tensor with type of float";
  CHECK_EQ((*in_attrs)[2], mshadow::kFloat32)
    << "the third input of `quantize` should be a tensor with type of float";
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 2, mshadow::kFloat32);
  return (*in_attrs)[0] != -1;
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_QUANTIZE_INL_H_
