#!/usr/bin/env python

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

# pylint: skip-file
import sys
sys.path.insert(0, "../../python/")
import mxnet as mx
import numpy as np
import numpy.random as rnd
from mxnet.test_utils import assert_almost_equal

def check_diff_to_scalar(A, x, rank=None):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0), (rank, A.asnumpy(), x)

# setup
keys = ['3', '5', '7']

rate = 2
shape = (2, 3)
big_shape = (1200, 1200)        # bigger than MXNET_KVSTORE_BIGARRAY_BOUND

kv = mx.kv.create('dist_sync')

def init_kv():
    # init kv dns keys
    # kv.init(keys, [mx.nd.ones(shape)] * len(keys))
    # kv.init('99', mx.nd.ones(big_shape))
    # init kv row_sparse keys
    # kv.init(rsp_keys, [mx.nd.ones(shape).tostype('row_sparse')] * len(rsp_keys))
    # kv.init('100', mx.nd.ones(big_shape).tostype('row_sparse'))
    # worker info
    my_rank = kv.rank
    nworker = kv.num_workers
    # init updater on servers
    # kv.set_optimizer(mx.optimizer.create('test', rescale_grad=rate))
    return kv, my_rank, nworker

def init_kv_compressed(kv):
    kv.set_gradient_compression({'type': 'signum', 'beta':0.5, 'recompress_type':'majority'})
    # init kv compression keys
    kv.init('11221', mx.nd.zeros(big_shape))
    kv.init('1121', mx.nd.zeros(shape))
    # to test inactive mode
    kv.init('1122', mx.nd.ones(shape))
    return kv, 0.5

def test_sync_push_pull():
    kv, my_rank, nworker = init_kv()

    def check_compr_ones(kv, nworker):
        for k,s in [('1121', shape),('11221', big_shape)]:
            val = mx.nd.zeros(s)
            kv.pull(k, val)
            curval = val[0][0].asnumpy()[0]
            kv.push(k, mx.nd.ones(s))
            val2 = mx.nd.zeros(s)
            kv.pull(k, val2)
            newval = curval + 1
            check_diff_to_scalar(val2, newval)
            # residual = 0  again

    def check_compr_pull_before_push(kv):
        for k,s in [('1121', shape), ('11221', big_shape)]:#,('112221',irregular_shape), #, ('1122',shape)]:
            if k=='1122':
                # tests that GC is not used for init of a key
                val = mx.nd.zeros(s)
                kv.pull(k, val)
                check_diff_to_scalar(val, 1)
            else:
                val = mx.nd.ones(s)
                kv.pull(k, val)
                check_diff_to_scalar(val, 0)

    def check_compr_zero(kv):
        for k,s in [('1121', shape)]:#,('112221',irregular_shape),('11221', big_shape)]:
            kv.push(k, mx.nd.zeros(s))
            # to check that all are set to 0s
            val = mx.nd.ones(s)
            kv.pull(k, val)
            check_diff_to_scalar(val, 0)

    def check_compr_random(kv, nworker):
        # set a seed so all workers generate same data. knowing this helps
        # calculate expected value after pull
        # mx.random.seed(123)
        # rnd.seed(123)
        nrepeat = 3
        compr_random_keys_shapes = [('2121', shape)]#,('212221',irregular_shape),('21221', big_shape)]
        # use new keys so residual is 0 for calculation of expected
        for k,s in compr_random_keys_shapes:
            kv.init(k, mx.nd.zeros(s))
        for k,s in compr_random_keys_shapes:
            curr_residual = np.zeros(s)
            for l in range(nrepeat):
                orig_val = mx.nd.zeros(s)
                kv.pull(k, orig_val)

                grad = mx.nd.array(rnd.randn(s[0], s[1]))
                print(grad)
                # creates a copy because push changes grad because of assignment
                grad_cpy = mx.nd.array(grad)
                kv.push(k, grad)
                val = mx.nd.zeros(s)
                kv.pull(k, val)

                diff = val - orig_val

                print(kv.rank, diff.asnumpy())

    kv, beta = init_kv_compressed(kv)
    # check_compr_pull_before_push(kv)
    check_compr_ones(kv, nworker)
    check_compr_random(kv, nworker)
    print('worker ' + str(my_rank) + ' is done with compression tests')

if __name__ == "__main__":
    test_sync_push_pull()
