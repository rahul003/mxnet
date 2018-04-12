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
#include <bitset>
#include "../operator/mxnet_op.h"

namespace mxnet {
namespace kvstore {

//void log_array(std::string& name, )

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
  // for each byte
  float *compr_block = out + (out_byte_id >> 2);
  // start and end are indices in original grad array
  // by 4 into 32 = into 8
  const int start = out_byte_id << 3;
  const int end = (start + 8 <= original_size) ? start + 8 : original_size;
  // cast as char* to manipulate bits of float addresses
  unsigned char *block_ptr = reinterpret_cast < unsigned char * > (compr_block) + (out_byte_id & 3);
//  LOG(INFO) << (void*) block_ptr;
  // set to 0 by default
  *block_ptr = 0;
  float* res = residual + start;
  float* g = grad + start;
  uint8_t mask = 1U << 7;
  for (int i = start; i < end; i++) {
    // adds gradient to existing residual to get updated grad
    *res = (*res * beta) + (oneminusbeta * (*g++));

    // set bit to 1 if positive, else sets to 0
    if (*res++ >= 0) {
      *block_ptr |= mask;
    }
    mask >>= 1;
  }
//  LOG(INFO) << std::bitset<8>(*block_ptr).to_string() ;
}
};

template<typename xpu>
void QuantizeSignumKernelLaunch(mshadow::Stream<xpu> *s, const std::vector<mxnet::TBlob> &inputs,
                                const float beta) {
  CHECK_NE(inputs[0].Size(), inputs[2].Size());
  mxnet::op::mxnet_op::Kernel<quantize_signum, xpu>
  ::Launch(s,
           inputs[2].Size() * 4,         // number of calls is each byte of compressed array
           inputs[0].Size(),         // original size
           inputs[2].dptr<float>(),  // compressed array
           inputs[0].dptr<float>(),  // original array
           inputs[1].dptr<float>(),  // residual array
           beta, 1-beta);               // positive threshold

//  std::string compressed;
//  for(int i=0; i<inputs[2].Size(); i++) {
//    compressed += std::bitset<sizeof(float)*CHAR_BIT>(*reinterpret_cast<unsigned long*>(inputs[2].dptr<float>() + i)).to_string() + " ";
//  }
//  LOG(INFO) <<  compressed;
}


struct quantize_majority {
MSHADOW_XINLINE static void Map(int out_byte_id,
                                int original_size,
                                float *out,
                                int *intsum) {
  // compr_block needs to store 8 values
  // out_byte_id represents id for 8 values, divide it by 4 to get float id
  float *compr_block = out + (out_byte_id >> 2);
  // start and end are indices in intsum array
  // by 4 into 32 = into 8
  const int start = out_byte_id << 3;
  const int end = (start + 8 <= original_size) ? start + 8 : original_size;
  // cast as char* to manipulate bits of float addressesi
  // also increments compr_block by remainder of out_byte_id when divided by 4
  // this fetches appropriate block
  unsigned char *block_ptr = reinterpret_cast < unsigned char * > (compr_block) + (out_byte_id & 3);
  *block_ptr = 0;
  int* g = intsum + start;
  // this mask checks whether MSB is 1
  uint8_t mask = 1U << 7;
  for (int i = start; i < end; i++) {
    // if intsum is greater than 0, implies majority is positive
    // then set bit to 1
    if (*g++ >= 0) {
      *block_ptr |= mask;
    }
    mask >>= 1; // move mask to next bit
  }
}
};

template<typename xpu>
inline void QuantizeFromIntSumKernelLaunch(mshadow::Stream<xpu> *s,
                                           const std::vector<mxnet::TBlob> &inputs,
                                           const int original_size,
                                           const CompressionType type) {
  if (type == CompressionType::kMajority) {
    mxnet::op::mxnet_op::Kernel<quantize_majority, xpu>
    ::Launch(s,
             inputs[1].Size() * 4,         // one for each byte (upto 8 values)
             original_size,            // original size
             inputs[1].dptr<float>(),  // to compressed array
             inputs[0].dptr<int>());   // from int array
  } else {
    LOG(FATAL) << "Unsupported quantization";
  }
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
  const uint8_t mask = 1U << (7- (i & 7) );
  const uint8_t masked = *ch_ptr & mask;
//  LOG(INFO) << (void*) ch_ptr << " " << std::bitset<8>(mask) << " " << std::bitset<8>(*ch_ptr);
  // if bit at that position is 0, set outval to -1 else to 1
  *outval += (( masked == mask) * 2 ) - 1;
}
};

template<typename xpu>
void DequantizeSignumKernelLaunch(mshadow::Stream<xpu> *s,
                                  const std::vector<mxnet::TBlob> &inputs) {
  // TODO ensure inputs[1] is set to 0 if you want only dequantized value. else it accumulates to the given location
//  std::string compressed;
//  for(int i=0; i<inputs[0].Size(); i++) {
//    compressed += std::bitset<sizeof(float)*CHAR_BIT>(*reinterpret_cast<unsigned long*>(inputs[0].dptr<float>() + i)).to_string() + " ";
//  }
//  LOG(INFO) <<  compressed;

  MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, DType, {
    mxnet::op::mxnet_op::Kernel<dequantize_signum, xpu>
    ::Launch(s,
             inputs[1].Size(),         // original size
             inputs[1].dptr<DType>(),  // out array
             inputs[0].dptr<float>());  // compressed array
  });
//  std::string out;
//  for(int i=0; i<inputs[1].Size(); i++) {
//    out += std::to_string(*(inputs[1].dptr<float>() + i)) + " ";
//  }
//
//  LOG(INFO) << out;

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
    *outval += pos_threshold;
  } else if (masked == negmask) {
    // use posbits for mask as posbits are both 1s
    // then compare masked with negbits to see if only negbits were set
    *outval += neg_threshold;
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

// these gpu functions are defined in gradient_compression.cu
void Quantize2BitImpl(mshadow::Stream<mshadow::gpu> *s, const std::vector<mxnet::TBlob> &inputs,
                      const float threshold);
void QuantizeSignumImpl(mshadow::Stream<mshadow::gpu> *s, const std::vector<mxnet::TBlob> &inputs,
                        const float beta);
void QuantizeFromIntSumImpl(mshadow::Stream<mshadow::gpu> *s, const std::vector<mxnet::TBlob> &inputs,
                            const CompressionType type);

void Dequantize2BitImpl(mshadow::Stream<mshadow::gpu> *s, const std::vector<mxnet::TBlob> &inputs,
                        const void* threshold);

void DequantizeSignumImpl(mshadow::Stream<mshadow::gpu> *s,
                          const std::vector<mxnet::TBlob> &inputs);

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
                                   const CompressionType type) {
  QuantizeFromIntSumKernelLaunch(s, inputs, inputs[0].Size(), type);
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

}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET_KVSTORE_GRADIENT_COMPRESSION_INL_H_
