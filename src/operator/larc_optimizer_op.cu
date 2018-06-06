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
 * \file larc_optimizer_op.cu
 * \brief LARC Optimizer operators
 * \author Clement Fuji Tsang
 */
#include "./larc_optimizer_op-inl.h"
#include <cub/cub.cuh>

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(larc_sgd_update)
.set_attr<FCompute>("FCompute<gpu>", LARCUpdate<SGDUpdate<gpu> >);

NNVM_REGISTER_OP(larc_sgd_mom_update)
.set_attr<FCompute>("FCompute<gpu>", LARCUpdate<SGDMomUpdate<gpu> >);
// .set_attr<FComputeEx>("FComputeEx<gpu>", SGDMomAsyncLrUpdateEx<gpu>);

NNVM_REGISTER_OP(larc_mp_sgd_update)
.set_attr<FCompute>("FCompute<gpu>", LARCUpdate<MP_SGDUpdate<gpu> >);

NNVM_REGISTER_OP(larc_mp_sgd_mom_update)
.set_attr<FCompute>("FCompute<gpu>", LARCUpdate<MP_SGDMomUpdate<gpu> >);

}  // namespace op
}  // namespace mxnet
