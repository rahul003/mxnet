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

#include <gtest/gtest.h>
#include <mxnet/tensor_blob.h>
#include "../../src/operator/activation-inl.h"
#include "../include/test_op_runner.h"
#include "../include/test_core_op.h"

using namespace mxnet;

using kwargs_t = test::op::kwargs_t;


template<typename DType = float>
static void RunTimingTest(const bool isGPU,
                          const kwargs_t& op_kwargs,
                          const char *op_name,
                          const char *backward_op_name = COREOP_BWD_OP_NAME_VALUE_NONE) {
  const kwargs_t kwargs = test::op::CoreOpExecutor<DType>::ArgsWithOpName(
    op_kwargs, op_name, backward_op_name);

  // prime code and cache before the performance runs
  test::op::CoreOperatorRunner<DType> runner;
  runner.RunBidirectional(false, { {16000}, {16000}, {1003} }, kwargs, 1);

  // Do the performance runs
  std::vector<std::vector<TShape>> shapes;
  if (test::performance_run) {
    shapes = {
      { {16000}, {16000}, {1003} },
      { {16000}, {16000}, {1003} },
      { {16000}, {16000}, {1003} },
      { {16000}, {16000}, {1003} },
      { {16000}, {16000}, {1003} },
    };
  } else {
    shapes = {
      { {16000}, {16000}, {1003} },
      { {16000}, {16000}, {1003} },
    };
  }
  const char *pu = isGPU ? "GPU" : "CPU";
  for (const std::vector<TShape> &shape_vector : shapes) {
    runner.TimingTest(std::string(op_name) + " Operator " + pu, isGPU, false, kwargs,
                      2, 10, shape_vector);
  }
}

/*!
 * \brief Generic bidirectional sanity test
 */
TEST(QUANTIZE_2BIT_PERF, ExecuteBidirectional) {
  typedef float DType;
  test::op::CoreOperatorRunner<DType> runner;
  runner.RunBidirectional(false,
                          { {16000}, {16000}, {1003} },
                          test::op::CoreOpExecutor<DType>::ArgsWithOpName(
                            { {"pos_threshold", "0.5"},
                              {"neg_threshold", "0.5"} },
                            "_contrib_quantize_2bit", COREOP_BWD_OP_NAME_VALUE_NONE ), 1);
}

/*!
 * \brief ActivationOp timing test for CPU
 */
TEST(QUANTIZE_2BIT_PERF, TimingCPU) {
  RunTimingTest<float>(false,
                       { {"pos_threshold", "0.5"},
                         {"neg_threshold", "0.5"} },
                       "_contrib_quantize_2bit");
}

#if MXNET_USE_CUDA == 1
/*!
 * \brief ActivationOp timing test for GPU
 */
TEST(QUANTIZE_2BIT_PERF, TimingGPU) {
  RunTimingTest<float>(true,
                       { {"pos_threshold", "0.5"},
                       {"neg_threshold", "0.5"} },
                       "_contrib_quantize_2bit");
}}
#endif  // MXNET_USE_CUDA == 1

