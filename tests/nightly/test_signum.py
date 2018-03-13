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
import argparse
import math

def check_diff_to_scalar(A, x, rank=None):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0), (rank, A.asnumpy(), x)

rate = 1
shape = (2, 3)
big_shape = (1200, 1200)        # bigger than MXNET_KVSTORE_BIGARRAY_BOUND
keys_shapes = [('1121', shape)]#, ('121221', (10,3))]#, ('1122', big_shape)]
pullinit_test_key_shape = [('12', big_shape)]
kv = mx.kv.create('dist_sync')

def test_sync_push_pull(options):
    def init_kv(options):
        kv.set_gradient_compression({'type': 'signum', 'beta':options.beta, 'recompress_type':'majority'})
        kv.set_optimizer(mx.optimizer.create('test', rescale_grad=rate))
        # init kv compression key
        for k,s in keys_shapes:
            kv.init(k, mx.nd.zeros(s))
        for k,s in pullinit_test_key_shape:
            kv.init(k, mx.nd.ones(s))

    def check_compr_pull_before_push():
        for k,s in pullinit_test_key_shape:
            val = mx.nd.zeros(s)
            kv.pull(k, val)
            check_diff_to_scalar(val, 1)

    def check_compr_ones():
        for k,s in keys_shapes:
            val = mx.nd.zeros(s)
            kv.pull(k, val)
            curval = val[0][0].asnumpy()[0]
            kv.push(k, mx.nd.ones(s) * 0.4)
            val2 = mx.nd.zeros(s)
            kv.pull(k, val2)
            newval = curval + (1 * rate)
            check_diff_to_scalar(val2, newval)

    def check_compr_random(nrepeat, nworker):
        # set a seed so all workers generate same data. knowing this helps
        # calculate expected value after pull
        start_seed = 0
        for l in range(nrepeat):
            seeds = range(start_seed + nworker)
            start_seed += nworker
            grads = {}
            grads_cpy = {}
            signs = {}
            majority = {}
            num_majority = math.ceil(nworker/2)
            for k,s in keys_shapes:
                grads[k] = []
                signs[k] = np.zeros(s)
                majority[k] = np.zeros(s)
                for w in range(nworker):
                    mx.random.seed(seeds[w % len(seeds)])
                    rnd.seed(seeds[w % len(seeds)])
                    rand_arr = mx.nd.array(rnd.randn(s[0], s[1]))
                    grads[k].append(rand_arr)
                    sgn = np.sign(rand_arr.asnumpy())
                    sgn[sgn==0] = 1
                    signs[k] += sgn
                majority[k][signs[k] >= 0] = 1
                majority[k][signs[k] < 0] = -1
            
            # creates a copy because push changes grad because of assignment
            grads_cpy = dict(grads)
            for k,s in keys_shapes:
                orig_val = mx.nd.zeros(s)
                kv.pull(k, orig_val)
                # print(orig_val)
                kv.push(k, grads_cpy[k][kv.rank])
                val = mx.nd.zeros(s)
                kv.pull(k, val)
                diff = val - orig_val
                try:
                    assert_almost_equal(diff.asnumpy(), majority[k])
                except AssertionError:
                    if kv.rank == 0:
                        print('key:',k)
                        print('----------------')
                        print('grads:', grads[k])
                        print('----------------')
                        print('majority:', majority[k])
                        print('----------------')
                        print('signs:', signs[k])
                        print('----------------')
                        print('diff:', diff)
                        print('----------------')
                    sys.exit(1)
    
    init_kv(options)
    check_compr_pull_before_push()
    check_compr_ones()
    check_compr_random(options.nrepeat, options.nworkers)
    print('worker ' + str(kv.rank) + ' is done with signum compression tests')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test signum gradient compression')
    parser.add_argument('--nworkers', type=int, default=4)
    parser.add_argument('--beta', type=float, default=0)
    parser.add_argument('--nrepeat', type=int, default=1)
    opt = parser.parse_args()
    assert (opt.nworkers == kv.num_workers), ("num_workers given does not equal those launched", kv.num_workers, opt.nworkers)
    test_sync_push_pull(opt)
