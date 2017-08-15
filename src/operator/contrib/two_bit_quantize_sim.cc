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
 * \file two_bit_quantize_sim.cc
 * \brief
 */
#include "./two_bit_quantize_sim-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_contrib_quantize_2bit)
.describe(R"code(Quantize a input tensor using 2-bit compression.
)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(3)
.set_attr<nnvm::FInferShape>("FInferShape", Quantize2BitShape)
.set_attr<nnvm::FInferType>("FInferType", Quantize2BitType)
.set_attr<FCompute>("FCompute<cpu>", Quantize2BitCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_quantize_2bit"})
.add_argument("input", "NDArray-or-Symbol", "A ndarray/symbol of type `float32`")
.add_argument("neg_shreshold", "NDArray-or-Symbol", "The negative shreshold")
.add_argument("pos_shreshold", "NDArray-or-Symbol", "The positive shreshold");

}  // namespace op
}  // namespace mxnet
