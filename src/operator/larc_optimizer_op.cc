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
 * \file larc_optimizer_op.cc
 * \brief LARC Optimizer operators
 * \author Clement Fuji Tsang
 */
#include "./larc_optimizer_op-inl.h"
#include "./elemwise_op_common.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(LARCParam);
DMLC_REGISTER_PARAMETER(LARCMomParam);

NNVM_REGISTER_OP(larc_sgd_update)
MXNET_ADD_SPARSE_OP_ALIAS(larc_sgd_update)
.describe(R"code(Update function for Stochastic Gradient Descent (SDG) optimizer.

It updates the weights using::

 weight = weight - learning_rate * gradient

If weight is of ``row_sparse`` storage type,
only the row slices whose indices appear in grad.indices are updated::

 for row in gradient.indices:
     weight[row] = weight[row] - learning_rate * gradient[row]

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LARCParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<2, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<2, 1, false, true, false>)
.set_attr<FCompute>("FCompute<cpu>", LARCUpdate<SGDUpdate<cpu> >)
// .set_attr<FComputeEx>("FComputeEx<cpu>", LARCUpdateEx<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_arguments(LARCParam::__FIELDS__());

NNVM_REGISTER_OP(larc_sgd_mom_update)
MXNET_ADD_SPARSE_OP_ALIAS(larc_sgd_mom_update)
.describe(R"code(Momentum update function for Stochastic Gradient Descent (SDG) optimizer.

Momentum update has better convergence rates on neural networks. Mathematically it looks
like below:

.. math::

  v_1 = \alpha * \nabla J(W_0)\\
  v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
  W_t = W_{t-1} + v_t

It updates the weights using::

  v = momentum * v - learning_rate * gradient
  weight += v

Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.

If weight and grad are both of ``row_sparse`` storage type and momentum is of ``default`` storage type,
standard update is applied.

If weight, grad and momentum are all of ``row_sparse`` storage type,
only the row slices whose indices appear in grad.indices are updated (for both weight and momentum)::

  for row in gradient.indices:
      v[row] = momentum[row] * v[row] - learning_rate * gradient[row]
      weight[row] += v[row]

)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LARCMomParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<3, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<3, 1>)
//.set_attr<FInferStorageType>("FInferStorageType", StdOptStorageType<2, 1>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", LARCUpdate<SGDMomUpdate<cpu> >)
// .set_attr<FComputeEx>("FComputeEx<cpu>", SGDMomUpdateEx<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mom", "NDArray-or-Symbol", "Momentum")
.add_arguments(LARCMomParam::__FIELDS__());

NNVM_REGISTER_OP(larc_mp_sgd_update)
.describe("Updater function for multi-precision sgd optimizer")
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LARCParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<3, 1>)
.set_attr<nnvm::FInferType>("FInferType", MP_SGD_InferType<2, 1, 3>)
.set_attr<FCompute>("FCompute<cpu>", LARCUpdate<MP_SGDUpdate<cpu> >)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "gradient")
.add_argument("weight32", "NDArray-or-Symbol", "Weight32")
.add_arguments(LARCParam::__FIELDS__());

NNVM_REGISTER_OP(larc_mp_sgd_mom_update)
.describe("Updater function for multi-precision sgd optimizer")
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LARCMomParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<4, 1>)
.set_attr<nnvm::FInferType>("FInferType", MP_SGD_InferType<2, 1, 4>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", LARCUpdate<MP_SGDMomUpdate<cpu> >)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mom", "NDArray-or-Symbol", "Momentum")
.add_argument("weight32", "NDArray-or-Symbol", "Weight32")
.add_arguments(LARCMomParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
