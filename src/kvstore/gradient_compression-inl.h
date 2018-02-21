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
 * \file gradient_compression-inl.h
 * \author Rahul Huilgol
 * \brief Declares and defines functions used to quantize and dequantize data
 */
#ifndef MXNET_KVSTORE_GRADIENT_COMPRESSION_INL_H_
#define MXNET_KVSTORE_GRADIENT_COMPRESSION_INL_H_

#include <vector>
#include "../operator/mxnet_op.h"
#include "gradient_compression.h"
#include "../../mshadow/mshadow/base.h"

namespace mxnet {
namespace kvstore {

struct quantize_2bit {
  MSHADOW_XINLINE static void Map(int out_block_id,
                                  int original_size,
                                  float *out,
                                  float *grad,
                                  float *residual,
                                  const float neg_threshold,
                                  const float pos_threshold) {
    // this block contains the compressed representation of
    // upto 16 values starting from out_block_id*16
    float *compr_block = out + out_block_id;
    // init to 0
    *compr_block = 0;
    // start and end are indices in original grad array
    const int start = out_block_id << 4;
    const int end = (start + 16 <= original_size) ? start + 16 : original_size;
    // cast as char* to manipulate bits of float addresses
    char *block_ptr = reinterpret_cast < char * > (compr_block);
    // masks to set bits when value meets pos_threshold
    // 0xc0 is mask when value is to be represented by the first two bits in a char*
    // 0xc0 means first two bits are set to 11
    const uint8_t posbits[] = {0xc0, 0x30, 0x0c, 0x03};
    // masks to set bits when value meets neg_threshold
    const uint8_t negbits[] = {0x80, 0x20, 0x08, 0x02};
    for (int i = start; i < end; i++) {
      // adds offset to reach appropriate byte
      char *curr_byte = block_ptr + ((i - start) >> 2);
      // adds gradient to existing residual to get updated grad
      residual[i] += grad[i];
      if (residual[i] >= pos_threshold) {
        // set data to 11
        *curr_byte |= posbits[(i & 3)];
        // reduce residual by pos_threshold
        residual[i] -= pos_threshold;
      } else if (residual[i] <= neg_threshold) {
        // set data to 10
        *curr_byte |= negbits[(i & 3)];
        residual[i] -= neg_threshold;
      }
    }
  }
};

template<typename xpu>
void Quantize2BitKernelLaunch(mshadow::Stream<xpu> *s, const std::vector<mxnet::TBlob> &inputs,
                              const float threshold) {
  mxnet::op::mxnet_op::Kernel<quantize_2bit, xpu>
    ::Launch(s,
            inputs[2].Size(),         // compressed array size
            inputs[0].Size(),         // original size
            inputs[2].dptr<float>(),  // compressed array
            inputs[0].dptr<float>(),  // original array
            inputs[1].dptr<float>(),  // residual array
            -1 *threshold,            // negative threshold
            threshold);               // positive threshold
}

struct quantize_signum {
  MSHADOW_XINLINE static void Map(int out_byte_id,
                                  int original_size,
                                  float *out,
                                  float *grad,
                                  float *residual,
                                  const float beta, const float oneminusbeta) {
    float *compr_block = out + (out_byte_id >> 2);
    // start and end are indices in original grad array

    // by 4 into 32 = into 8
    const int start = out_byte_id << 3;
    const int end = (start + 8 <= original_size) ? start + 8 : original_size;
    // cast as char* to manipulate bits of float addresses
    unsigned char *block_ptr = reinterpret_cast < unsigned char * > (compr_block) + (out_byte_id & 3);
    *block_ptr = 0;
    float* res = residual + start;
    float* g = grad + start;
    uint8_t mask = 1U << 7;
    for (int i = start; i < end; i++) {
      // adds offset to reach appropriate byte
      // adds gradient to existing residual to get updated grad
      *res = (*res * beta) + (oneminusbeta * (*g++));
      if (*res++ >= 0) {
        *block_ptr |= mask;
      }
      mask >>= 1;
    }
  }
};

template<typename xpu>
void QuantizeSignumKernelLaunch(mshadow::Stream<xpu> *s, const std::vector<mxnet::TBlob> &inputs,
                              const float beta) {
  mxnet::op::mxnet_op::Kernel<quantize_signum, xpu>
    ::Launch(s,
            inputs[2].Size() * 4,         // compressed array size
            inputs[0].Size(),         // original size
            inputs[2].dptr<float>(),  // compressed array
            inputs[0].dptr<float>(),  // original array
            inputs[1].dptr<float>(),  // residual array
            beta, 1-beta);               // positive threshold
}

struct dequantize_signum {
  template <typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType *out,
                                  float *in) {
    // get position of dequantized value to fill
    DType *outval = out + i;
    // gets byte which holds quantized value for this position
    unsigned char *ch_ptr = reinterpret_cast<unsigned char *>(in + (i >> 5));
    ch_ptr += ((i & 31) >> 3);

    // if bit set at that position is 0, set outval to -1 else to 1
    *outval += ((*ch_ptr & (1U << (7 - ( i & 7 )))) * 2 ) - 1;
  }
};

template<typename xpu>
void DequantizeSignumKernelLaunch(mshadow::Stream<xpu> *s,
                                  const std::vector<mxnet::TBlob> &inputs) {
  // TODO ensure inputs[1] is set to 0 if you want only dequantized value. else it accumulates to the given location
  MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, DType, {
    mxnet::op::mxnet_op::Kernel<dequantize_signum, xpu>
    ::Launch(s,
             inputs[1].Size(),         // original size
             inputs[1].dptr<DType>(),  // out array
             inputs[0].dptr<float>());  // compressed array
  });
}

struct dequantize_2bit {
  template<typename DType, typename FType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType *out,
                                  FType *in,
                                  const DType neg_threshold,
                                  const DType pos_threshold) {
    // get position of dequantized value to fill
    DType *outval = out + i;
    // gets byte which holds quantized value for this position
    char *ch_ptr = reinterpret_cast<char *>(in + (i >> 4));
    ch_ptr += ((i & 15) >> 2);
    // masks used to quantize data
    const uint8_t posbits[] = {0xc0, 0x30, 0x0c, 0x03};
    const uint8_t negbits[] = {0x80, 0x20, 0x08, 0x02};
    // col denotes which two bits of a byte are set for this value
    // col=0 implies first two bits, col=3 implies last two bits,...
    const int col = i & 3;
    const uint8_t mask = posbits[col];
    const uint8_t negmask = negbits[col];
    const uint8_t masked = *ch_ptr & mask;
    if (masked == mask) {
      *outval = pos_threshold;
    } else if (masked == negmask) {
      // use posbits for mask as posbits are both 1s
      // then compare masked with negbits to see if only negbits were set
      *outval = neg_threshold;
    } else {
      *outval = 0;
    }
  }
};

template<typename xpu>
void Dequantize2BitKernelLaunch(mshadow::Stream<xpu> *s,
                                const std::vector<mxnet::TBlob> &inputs,
                                const void *threshold) {
  MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, DType, {
    const DType th = *static_cast<const DType *>(threshold);
    mxnet::op::mxnet_op::Kernel<dequantize_2bit, xpu>
    ::Launch(s,
             inputs[1].Size(),         // original size
             inputs[1].dptr<DType>(),      // out array
             inputs[0].dptr<float>(),  // compressed array
             static_cast<DType>(th * -1), // negative threshold
             th);               // positive threshold
  });
}

struct quantize_logk {
  MSHADOW_XINLINE static void Map(int out_block_id,   // id of parallel kernel
                                  int num_blocks,
                                  int block_size,     // number of output floats to process by each call
                                  int original_size,
                                  int num_workers,
                                  int num_bits,
                                  float *out,         // size = num_blocks * block_size
                                  int *sum) {
    // this call is responsible for quantizing values into block_size number of 32bit locations
    // if `b` bits are being used for a single gradient value, then it is guaranteed that
    // `32*block_size` is divisible by `b`. so the number of values in the original sum array that this call
    // is responsible for is equal to `32 * block_size / b`

    float *compr_float = out + (out_block_id * block_size);
    // number of sum values to be processed by this kernel call are
    // (out_block_id * block_size)/num_bits

    bool is_last = (out_block_id == num_blocks - 1);
    int num_prev_elems = 0;
    if (out_block_id != 0) {
      num_prev_elems = (out_block_id - 1) * block_size;
    }

    for (int i = 0; i < block_size; i++, compr_float++) {
      *compr_float = 0;

      // inclusive
      int st_pos = 0;
      // including
      int end_pos = num_bits - 1;

      // this will be incremented four times to cover the full float
      unsigned char *byte_ptr = reinterpret_cast < unsigned char * > (compr_float);
      while ((end_pos < block_size * 32) &&
             (!is_last || (is_last && num_prev_elems < original_size ))) {
        uint8_t s = (uint8_t) *(sum++);
        num_prev_elems++;

        if (st_pos / 8 == end_pos / 8) {
          // doesn't cross byte boundary
          *(byte_ptr) |= s << (8 - num_bits);
        } else {
          int bits_remaining = 8 - (end_pos % 8 + 1);
          // shift by num_bits - bits_remaining, so that bits_remaining is when the value starts
          *(byte_ptr++) |= s >> (num_bits - bits_remaining);
          *(byte_ptr) |= s << (8 - num_bits + bits_remaining);
        }
        st_pos += num_bits;
        end_pos += num_bits;

        if (st_pos % 8 == 0) byte_ptr++;
      }
    }
  }
};

struct dequantize_logk {
  MSHADOW_XINLINE static void Map(int compr_block_id,   // id of parallel kernel
                                  int num_blocks,
                                  int block_size,     // number of compressed floats to process by each call
                                  int original_size,
                                  int num_workers,
                                  int num_bits,
                                  float threshold,
                                  float *out,
                                  float *compr) {
    //TODO check endianness
    float *compr_float = compr + compr_block_id;
    bool is_last = (compr_block_id == num_blocks - 1);
    int num_prev_elems = 0;
    if (compr_block_id != 0) {
      num_prev_elems = (compr_block_id - 1) * block_size;
    }

    // inclusive
    int st_pos = 0;
    // including
    int end_pos = num_bits - 1;
    uint8_t bitmask = (0x01 << num_bits) - 1;
    for (int i = 0; i < block_size; i++, compr_float++) {
      *compr_float = 0;
      unsigned char *byte_ptr = reinterpret_cast < unsigned char * > (compr_float);
      while ((end_pos < block_size * 32) &&
             (!is_last || (is_last && num_prev_elems < original_size ))) {
        if (end_pos / 8 == st_pos / 8) {
          int curval = *byte_ptr;
          curval >>= (8 - (end_pos % 8) - 1);
          curval &= bitmask;
          *(out++) = ( curval - num_workers) * threshold;
        } else {
//          uint8_t curval = ((*byte_ptr) >> (8 - (end_pos % 8) - 1));
          uint8_t num_bits_overflowed = (end_pos + 1) % 8;
          // left shift bits into position
          uint8_t curval = *(byte_ptr++) << num_bits_overflowed;
          curval &= bitmask;
          // now bring next byte bits here
          // TODO confirm byteptr is unaffected
          curval |= (*byte_ptr >> (8 - num_bits_overflowed));
          *(out++) = (curval - num_workers) * threshold;
        }
        num_prev_elems++;

        st_pos += num_bits;
        end_pos += num_bits;

        if (st_pos % 8 == 0) byte_ptr++;
      }
    }
  }
};

struct quantize_majority {
  MSHADOW_XINLINE static void Map(int out_byte_id,
                                  int original_size,
                                  int num_majority,
                                  float *out,
                                  int *intsum) {
    float *compr_block = out + (out_byte_id >> 2);
    // start and end are indices in original grad array
    // by 4 into 32 = into 8
    const int start = out_byte_id << 3;
    const int end = (start + 8 <= original_size) ? start + 8 : original_size;
    // cast as char* to manipulate bits of float addresses
    unsigned char *block_ptr = reinterpret_cast < unsigned char * > (compr_block) + (out_byte_id & 3);
    *block_ptr = 0;
    int* g = intsum + start;
    uint8_t mask = 1U << 7;
    for (int i = start; i < end; i++) {
      if (*g++ >= num_majority) {
        *block_ptr |= mask;
      }
      mask >>= 1;
    }
  }
};

template<typename xpu>
inline void QuantizeFromIntSumKernelLaunch(mshadow::Stream<xpu> *s,
                                           const std::vector<mxnet::TBlob> &inputs,
                                           const int num_workers,
                                           const int original_size,
                                           const CompressionType type) {
  if (type == CompressionType::kLogK) {
    int block_size = lcm(num_workers, 32) / 32;
    // each block is responsible for block_size number of floats into which compressed data is packed

    // number of bits used for each value
    int num_bits = (int) ceil(log2(float(2 * num_workers + 1)));

    if (num_bits > 8) {
      LOG(FATAL) << "Gradient compression unsupported for this type and number of workers right now";
    }
    mxnet::op::mxnet_op::Kernel<quantize_logk, xpu>
    ::Launch(s,
             (inputs[1].Size()) / block_size,   // number of parallel kernels, one for each block
             (inputs[1].Size()) / block_size, // number of blocks
             block_size,               // number of output floats (32bits) to process for each kernel call
             original_size,            // original size
             num_workers,              // num_workers
             num_bits,                 // num_bits
             inputs[1].dptr<float>(),  // to compressed array
             inputs[0].dptr<int>());   // from int array
  } else if (type == CompressionType::kMajority) {
    int num_majority = int(ceil((float) num_workers / 2));
    mxnet::op::mxnet_op::Kernel<quantize_majority, xpu>
    ::Launch(s,
             inputs[1].Size() * 4,         // compressed size
             original_size,            // original size
             num_majority,              // num_workers
             inputs[1].dptr<float>(),  // to compressed array
             inputs[0].dptr<int>());   // from int array
  } else {
    LOG(FATAL) << "Unsupported quantization";
  }
}
template<typename xpu>
inline void DequantizeKernelLaunch(mshadow::Stream<xpu> *s,
                             const std::vector<mxnet::TBlob> &inputs,
                             const float threshold,
                             const int num_workers,
                             const int original_size, const CompressionType type) {

  int block_size = lcm(num_workers, 32) / 32;

  // number of bits used for each value
  int num_bits = (int) ceil(log2(float(2 * num_workers + 1)));
  CHECK_LE(num_bits, 8);

  if (type == CompressionType::kLogK) {
    mxnet::op::mxnet_op::Kernel<dequantize_logk, xpu>
    ::Launch(s,
             (inputs[0].Size())/block_size,   // number of parallel kernels, one for each block
             (inputs[0].Size())/block_size,   // number of blocks
             block_size,               // number of output floats (32bits) to process for each kernel call
             original_size,            // original size
             num_workers,              // num_workers
             num_bits,                 // num_bits
             threshold,
             inputs[1].dptr<float>(),  // original sized array
             inputs[0].dptr<float>());   // compressed array
  } else {
    LOG(FATAL) << "Unsupported dequantization";
  }

}

// TODO merge below and verify compilation on gpu
// these gpu functions are defined in gradient_compression.cu
void Quantize2BitImpl(mshadow::Stream<mshadow::gpu> *s, const std::vector<mxnet::TBlob> &inputs,
                      const float threshold);
void QuantizeSignumImpl(mshadow::Stream<mshadow::gpu> *s, const std::vector<mxnet::TBlob> &inputs,
                        const float beta);
void QuantizeFromIntSumImpl(mshadow::Stream<mshadow::gpu> *s, const std::vector<mxnet::TBlob> &inputs,
                            const int num_workers);

void Dequantize2BitImpl(mshadow::Stream<mshadow::gpu> *s, const std::vector<mxnet::TBlob> &inputs,
                        const void* threshold);

void DequantizeSignumImpl(mshadow::Stream<mshadow::gpu> *s,
                          const std::vector<mxnet::TBlob> &inputs);

void DequantizeImpl(mshadow::Stream<mshadow::gpu> *s, const std::vector<mxnet::TBlob> &inputs,
                        const float threshold, const int num_workers, const CompressionType type);

inline void Quantize2BitImpl(mshadow::Stream<mshadow::cpu> *s,
                             const std::vector<mxnet::TBlob> &inputs,
                             const float threshold) {
  Quantize2BitKernelLaunch(s, inputs, threshold);
}

inline void QuantizeSignumImpl(mshadow::Stream<mshadow::cpu> *s,
                             const std::vector<mxnet::TBlob> &inputs,
                             const float beta) {
  QuantizeSignumKernelLaunch(s, inputs, beta);
}

inline void QuantizeFromIntSumImpl(mshadow::Stream<mshadow::cpu> *s,
                                   const std::vector<mxnet::TBlob> &inputs,
                                   const int num_workers,
                                   const CompressionType type) {
  QuantizeFromIntSumKernelLaunch(s, inputs, num_workers, inputs[0].Size(), type);
}

inline void Dequantize2BitImpl(mshadow::Stream<mshadow::cpu> *s,
                               const std::vector<mxnet::TBlob> &inputs,
                               const void *threshold) {
  Dequantize2BitKernelLaunch(s, inputs, threshold);
}


inline void DequantizeSignumImpl(mshadow::Stream<mshadow::cpu> *s,
                                 const std::vector<mxnet::TBlob> &inputs) {
    DequantizeSignumKernelLaunch(s, inputs);
}

inline void DequantizeImpl(mshadow::Stream<mshadow::cpu> *s,
                               const std::vector<mxnet::TBlob> &inputs,
                               const float threshold,
                               const int num_workers,
                               const CompressionType type) {
  DequantizeKernelLaunch(s, inputs, threshold, num_workers, inputs[1].Size(), type);
}

}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET_KVSTORE_GRADIENT_COMPRESSION_INL_H_
