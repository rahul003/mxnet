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
irregular_shape = (1241, 1211)
big_shape = (1200, 1200)        # 0bigger than MXNET_KVSTORE_BIGARRAY_BOUND
keys_shapes = [('1', irregular_shape)]#, ('2', shape), ('3', irregular_shape)]
kv = mx.kv.create('dist_sync')

def test_sync_push_pull(options):
    def init_kv(options):
        kv.set_gradient_compression({'type': 'signum',
                                     'beta': options.beta,
                                     'server_compression_type': options.recompress_type})
        kv.set_optimizer(mx.optimizer.create('test', rescale_grad=rate))
        # init kv compression key
        for k,s in keys_shapes:
            kv.init(k, mx.nd.zeros(s))

    def check_compr_pushes(options, value):
        for i in range(options.nrepeat):
            for k, s in keys_shapes:
                val = mx.nd.zeros(s)
                kv.pull(k, val)
                curval = val[0][0].asnumpy()[0]
                kv.push(k, mx.nd.ones(s) * value)
                val2 = mx.nd.zeros(s)
                kv.pull(k, val2)
                sign = 1 if value >= 0 else -1
                if options.recompress_type == 'majority':
                    newval = curval + (1 * rate * sign)
                else:
                    newval = curval + (kv.num_workers * rate * sign)
                check_diff_to_scalar(val2, newval)

    def check_compr_pos(options):
        check_compr_pushes(options, 0.4)

    def check_compr_neg(options):
        check_compr_pushes(options, -0.4)

    def check_compr_random_majority(nrepeat, nworker):
        # set a seed so all workers generate same data. knowing this helps
        # calculate expected value after pull
        start_seed = 0
        for l in range(nrepeat):
            seeds = range(start_seed + nworker)
            start_seed += nworker
            grads = {}
            signs = {}
            majority = {}
            for k, s in keys_shapes:
                grads[k] = []
                signs[k] = np.zeros(s)
                majority[k] = np.zeros(s)
                for w in range(nworker):
                    mx.random.seed(seeds[w % len(seeds)])
                    rnd.seed(seeds[w % len(seeds)])
                    rand_arr = mx.nd.array(rnd.randn(s[0], s[1]))
                    grads[k].append(rand_arr)
                    sgn = np.sign(rand_arr.asnumpy())
                    sgn[sgn == 0] = 1
                    signs[k] += sgn
                majority[k][signs[k] >= 0] = 1
                majority[k][signs[k] < 0] = -1
            
            # creates a copy because push changes grad because of assignment
            grads_cpy = dict(grads)
            for k, s in keys_shapes:
                orig_val = mx.nd.zeros(s)
                kv.pull(k, orig_val)
                # print(orig_val)
                kv.push(k, grads_cpy[k][kv.rank])
                val = mx.nd.zeros(s)
                kv.pull(k, val)
                diff = val - orig_val
                try:
                    assert_almost_equal(diff.asnumpy(), majority[k])
                except AssertionError as e:
                    if kv.rank == 0:
                        e.args += ('key:', k, 'grads', grads[k], 'majority', majority[k], 'signs', signs[k], 'diff', diff)

    def check_compr_random_none(nrepeat, nworker):
        # set a seed so all workers generate same data. knowing this helps
        # calculate expected value after pull
        start_seed = 0
        for l in range(nrepeat):
            seeds = range(start_seed + nworker)
            start_seed += nworker
            grads = {}
            signs = {}
            for k, s in keys_shapes:
                grads[k] = []
                signs[k] = np.zeros(s)
                for w in range(nworker):
                    mx.random.seed(seeds[w % len(seeds)])
                    rnd.seed(seeds[w % len(seeds)])
                    rand_arr = mx.nd.array(rnd.randn(s[0], s[1]))
                    grads[k].append(rand_arr)
                    sgn = np.sign(rand_arr.asnumpy())
                    sgn[sgn == 0] = 1
                    signs[k] += sgn

            # creates a copy because push changes grad because of assignment
            grads_cpy = dict(grads)
            for k, s in keys_shapes:
                orig_val = mx.nd.zeros(s)
                kv.pull(k, orig_val)
                kv.push(k, grads_cpy[k][kv.rank])
                val = mx.nd.zeros(s)
                kv.pull(k, val)
                diff = val - orig_val
                # print(grads_cpy[k][kv.rank], diff)
                try:
                    assert_almost_equal(diff.asnumpy(), signs[k])
                except AssertionError as e:
                    if kv.rank == 0:
                        e.args += ('key:', k, 'grads', grads[k], 'signs', signs[k], 'diff', diff)

    init_kv(options)
    check_compr_pos(options)
    # check_compr_neg(options)
    # if options.recompress_type == 'majority':
    #     check_compr_random_majority(options.nrepeat, kv.num_workers)
    # elif options.recompress_type == 'none':
    #     check_compr_random_none(options.nrepeat, kv.num_workers)
    # print('worker ' + str(kv.rank) + ' is done with signum compression tests')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test signum gradient compression')
    parser.add_argument('--beta', type=float, default=0)
    parser.add_argument('--nrepeat', type=int, default=1)
    parser.add_argument('--recompress-type', type=str, default='none')
    opt = parser.parse_args()
    test_sync_push_pull(opt)
