#!/bin/bash

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

set -e
runme() {
	cmd=$*
	echo "$cmd"
	$cmd
	ret=$?
	if [[ ${ret} != 0 ]]; then
		echo " "
		echo "ERROR: Return value non-zero for: $cmd"
		echo " "
		exit 1
	fi
}

sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get -y install time g++-5
runme make clean >/dev/null
runme mkdir build
echo "Starting make on CPU with g++5"
runme /usr/bin/time -f "%e" CC=gcc-5 CXX=g++-5 make USE_OPENCV=1 USE_BLAS=openblas -j $(nproc) 2>&1 | tee build/cpu_compile_log.txt
echo "##############################"
echo "Starting make on GPU with g++5"
runme /usr/bin/time -f "%e" CC=gcc-5 CXX=g++-5 make USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 -j $(nproc) 2>&1 | tee build/gpu_compile_log.txt
echo "Finished make. Now processing output"
python tests/nightly/compilation_warnings/process_output.py cpu build/cpu_compile_output.txt
python tests/nightly/compilation_warnings/process_output.py gpu build/gpu_compile_output.txt
