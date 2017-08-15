# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
import numpy as np
from log_util import LogUtil
class STTModule(mx.mod.Module):
# 1-bit compression
    def Gradient_Compression_1bit(self):
        scale = 0.01
        log = LogUtil().getlogger()
        array = self._exec_group.grad_arrays
        num_params = len(array)
        if not hasattr(self,'tmp_array'):
            log.info(array[0][0])
            log.info(array[0][0].shape)
            log.info(array[0][0].asnumpy().shape)
            self.tmp_array = []
            # [np.zeros(array[i][j].asnumpy().shape) for i in range(num_params)] for j in range(len(self._context))]
            for i in range(num_params):
                t = []
                for c in range(len(self._context)):
                    t.append(np.zeros(array[i][c].asnumpy().shape))
                self.tmp_array.append(t)
            # for i in range(num_params):
            #     for c in range(len(self._context)):
            #         log.info(self.tmp_array[i][c].shape)
            log.info('created temp array for 1bit compression')        
        for i in range(num_params):
            for c in range(len(self._context)):
                sum_pos = 0
                sum_nega = 0
                pos_num = 0
                nega_num = 0
                layer = array[i][c].asnumpy()
                # Get average value of positive and negative
                layer += self.tmp_array[i][c]
                # log.info(layer.shape)
                sum_pos += sum(layer[layer>=0])
                sum_nega += sum(layer[layer<0])
                pos_num += layer[layer>=0].size
                nega_num += layer[layer<0].size
                # Write average value back
                # add res
                self.tmp_array[i][c] = np.copy(layer)

                layer[layer>=0] = sum_pos / (pos_num+scale)
                layer[layer<0] = sum_nega / (nega_num+scale)
                # calc res
                self.tmp_array[i][c] -= layer
                mx.nd.array(layer).copyto(self._exec_group.grad_arrays[i][c])

    def compress(self, numbits):
        if numbits==1:
            self.Gradient_Compression_1bit()

class STTBucketingModule(mx.mod.BucketingModule):
    def save_checkpoint(self, prefix, epoch, save_optimizer_states=False):
        symbol, data_names, label_names = self._sym_gen(self._default_bucket_key)
        symbol.save('%s-symbol.json' % prefix)
        param_name = '%s-%04d.params' % (prefix, epoch)
        self.save_params(param_name)
        if save_optimizer_states:
            state_name = '%s-%04d.states' % (prefix, epoch)
            self._curr_module.save_optimizer_states(state_name)
