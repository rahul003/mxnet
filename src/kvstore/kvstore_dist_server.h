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
 * Copyright (c) 2015 by Contributors
 * \file mxnet_kvstore_dist_server.h
 * \brief parameter server for distributed kvstore
 */
#ifndef MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
#define MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
#include <queue>
#include <string>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <functional>
#include <future>
#include <vector>
#include "ps/ps.h"
#include "mxnet/kvstore.h"
#include "../operator/tensor/elemwise_binary_op-inl.h"
#include "../operator/tensor/init_op.h"

namespace mxnet {
namespace kvstore {

// maintain same order in frontend.
enum class CommandType {
  kController, kSetMultiPrecision, kStopServer, kSyncMode, kSetGradientCompression,
};

enum class RequestType {
  kDefaultPushPull, kRowSparsePushPull, kCompressedPush, kCompressedPull, kCompressedFullPull
};

struct DataHandleType {
  RequestType requestType;
  int dtype;
};

/*!
 * Uses Cantor pairing function to generate a unique number given two numbers.
 * This number can also be inverted to find the unique pair whose Cantor value is this number.
 * Ref: https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
 * \param requestType RequestType
 * \param dtype integer
 * \return Cantor value of arguments
 */
static int GetCommandType(RequestType requestType, int d) {
  int m = static_cast<int>(requestType);
  return (((m + d) * (m + d + 1)) / 2) + d;
}

/*!
 * Unpairs Cantor value and finds the two integers used to pair.
 * Then returns DataHandleType object with those numbers.
 * \param cmd DataHandleCommand generated by GetCommandType function
 * \return DataHandleType
 */
static DataHandleType DepairDataHandleType(int cmd) {
  int w = std::floor((std::sqrt(8 * cmd + 1) - 1)/2);
  int t = ((w * w) + w) / 2;
  int y = cmd - t;
  int x = w - y;
  CHECK_GE(x, 0);
  CHECK_GE(y, 0);
  DataHandleType type;
  type.requestType = static_cast<RequestType>(x);
  type.dtype = y;
  return type;
}

/**
 * \brief executor runs a function using the thread called \ref Start
 */
class Executor {
 public:
  /**
   * \brief start the executor
   */
  void Start() {
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
      cond_.wait(lk, [this]{return !queue_.empty();});
      Block blk = std::move(queue_.front());
      queue_.pop();
      lk.unlock();

      if (blk.f) {
        blk.f(); blk.p->set_value();
      } else {
        blk.p->set_value(); break;
      }
      lk.lock();
    }
  }

  /**
   * \brief function
   */
  typedef std::function<void()> Func;

  /**
   * \brief let the thread called \ref Start to exec a function. threadsafe
   */
  void Exec(const Func& func) {
    Block blk(func);
    auto fut = blk.p->get_future();
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push(std::move(blk));
      cond_.notify_one();
    }
    fut.wait();
  }

  /**
   * \brief stop the thread, threadsafe
   */
  void Stop() {
    Exec(Func());
  }

 private:
  struct Block {
  explicit Block(const Func& func) : f(func), p(std::make_shared<std::promise<void>>()) { }
    Func f;
    std::shared_ptr<std::promise<void>> p;
  };
  std::queue<Block> queue_;
  std::mutex mu_;
  std::condition_variable cond_;
};

class KVStoreDistServer {
 public:
  KVStoreDistServer() {
    using namespace std::placeholders;
    ps_server_ = new ps::KVServer<char>(0);
    static_cast<ps::SimpleApp*>(ps_server_)->set_request_handle(
        std::bind(&KVStoreDistServer::CommandHandle, this, _1, _2));
    ps_server_->set_request_handle(
        std::bind(&KVStoreDistServer::DataHandleEx, this, _1, _2, _3));
    sync_mode_ = false;
    gradient_compression_ = std::make_shared<GradientCompression>();
    log_verbose_ = dmlc::GetEnv("MXNET_KVSTORE_DIST_VERBOSE", false);
  }

  ~KVStoreDistServer() {
    delete ps_server_;
  }

  void set_controller(const KVStore::Controller& controller) {
    CHECK(controller);
    controller_ = controller;
  }

  void set_updater(const KVStore::Updater& updater)  {
    CHECK(updater);
    updater_ = updater;
  }

  /**
   * \brief blocked until received the command \a kSyncMode
   */
  void Run() {
    exec_.Start();
  }

 private:
  struct UpdateBuf {
    std::vector<ps::KVMeta> request;
    NDArray merged;
    // temp_array is used to cast received values as float32 for computation if required
    NDArray temp_array;
    // int_array is used to sum up compressed gradients during gradient compression
    NDArray int_array;
    // recompressed array if server_compression is not none during gradient compression
    NDArray requantized;
  };

  void CommandHandle(const ps::SimpleData& recved, ps::SimpleApp* app) {
    CommandType recved_type = static_cast<CommandType>(recved.head);
    if (recved_type == CommandType::kStopServer) {
      exec_.Stop();
    } else if (recved_type == CommandType::kSyncMode) {
      sync_mode_ = true;
    } else if (recved_type == CommandType::kSetGradientCompression) {
      gradient_compression_->DecodeParams(recved.body);
    } else if (recved_type == CommandType::kSetMultiPrecision) {
      // uses value 1 for message id from frontend
      if (!multi_precision_) {
        multi_precision_ = true;
        CreateMultiPrecisionCopies();
      }
    } else if (recved_type == CommandType::kController) {
      // value of 0
      // let the main thread to execute ctrl, which is necessary for python
      exec_.Exec([this, recved]() {
          CHECK(controller_);
          controller_(recved.head, recved.body);
        });
    } else {
      LOG(FATAL) << "Unknown command type received " << recved.head;
    }
    app->Response(recved);
  }

  /*
   * For keys already initialized, if necessary create stored_realt.
   * This will only be used if by some wrong usage of kvstore,
   * some keys are initialized before optimizer is set.
   */
  void CreateMultiPrecisionCopies() {
    for (auto const& stored_entry : store_) {
      const int key = stored_entry.first;
      const NDArray& stored = stored_entry.second;
      if (stored.dtype() != mshadow::kFloat32) {
        auto& stored_realt = store_realt_[key];
        if (stored.storage_type() == kRowSparseStorage) {
          stored_realt = NDArray(kRowSparseStorage, stored.shape(), stored.ctx(),
                                 true, mshadow::kFloat32);
        } else {
          stored_realt = NDArray(stored.shape(), stored.ctx(), false, mshadow::kFloat32);
        }

        auto& update = update_buf_[key];
        if (!update.merged.is_none()) {
          if (update.merged.storage_type() == kRowSparseStorage) {
            update.merged = NDArray(kRowSparseStorage, update.merged.shape(), update.merged.ctx(),
                                    true, mshadow::kFloat32);
          } else {
            update.merged = NDArray(update.merged.shape(), update.merged.ctx(), false,
                                    mshadow::kFloat32);
          }
        }
        CHECK(update.request.size() == 0)
          << ps::MyRank() << "Multiprecision mode can not be set while pushes are underway."
          << "Please set optimizer before pushing keys." << key << " " << update.request.size();

        CopyFromTo(stored, stored_realt);
      }
    }
    for (auto const& stored_realt_entry : store_realt_) {
      stored_realt_entry.second.WaitToRead();
    }
  }

  void DataHandleEx(const ps::KVMeta& req_meta,
                    const ps::KVPairs<char>& req_data,
                    ps::KVServer<char>* server) {
    DataHandleType type = DepairDataHandleType(req_meta.cmd);
    switch (type.requestType) {
      case RequestType::kRowSparsePushPull:
        DataHandleRowSparse(type, req_meta, req_data, server);
        break;
      case RequestType::kCompressedPush:
      case RequestType::kCompressedPull:
      case RequestType::kCompressedFullPull:
        DataHandleCompressed(type, req_meta, req_data, server);
        break;
      case RequestType::kDefaultPushPull:
        DataHandleDefault(type, req_meta, req_data, server);
        break;
    }
  }

  inline bool has_multi_precision_copy(const DataHandleType type) {
    return multi_precision_ && type.dtype != mshadow::kFloat32;
  }

  inline void ApplyUpdates(const DataHandleType type, const int key,
                           UpdateBuf *update_buf, ps::KVServer<char>* server, bool reset_merged = false) {
    if (!sync_mode_ || update_buf->request.size() == (size_t) ps::NumWorkers()) {
      // let the main thread to execute updater_, which is necessary for python
      auto& stored = has_multi_precision_copy(type) ? store_realt_[key] : store_[key];
      auto& update =  sync_mode_ ? update_buf->merged : update_buf->temp_array;
      if (updater_) {
        exec_.Exec([this, key, &update, &stored](){
          CHECK(updater_);
          updater_(key, update, &stored);
        });
      } else {
        CHECK(sync_mode_) << "Updater needs to be set for async mode";
        // if no updater, just copy
        CopyFromTo(update_buf->merged, &stored);
      }

      if (log_verbose_)  {
        LOG(INFO) << "sent response to " << update_buf->request.size() << " workers";
      }
      for (const auto& req : update_buf->request) {
        server->Response(req);
      }
      update_buf->request.clear();
      if (has_multi_precision_copy(type)) CopyFromTo(stored, store_[key]);
      stored.WaitToRead();
      // reset needed for gradient compression
      if (reset_merged) update_buf->merged = 0;
    } else {
      update_buf->merged.WaitToRead();
    }
  }

  /*!
   * Used when server recompresses gradients, in this case requantized aggregation of gradients is what server
   * maintains for each key
   * @param key
   * @param update_buf
   * @param server
   * @param original_size
   */
  inline void RecompressUpdates(const int key, UpdateBuf *update_buf, ps::KVServer<char>* server,
                                int original_size) {
    CHECK(gradient_compression_->get_server_compression_type() != CompressionType::kNone);
    CHECK(sync_mode_);
    if (update_buf->request.size() == (size_t) ps::NumWorkers()) {
      gradient_compression_->Requantize(update_buf->int_array, &(update_buf->requantized), 0);
      for (const auto &req : update_buf->request) {
        server->Response(req);
      }
      update_buf->request.clear();
      update_buf->requantized.WaitToRead();
      // reset int array for gradient compression
      update_buf->int_array = 0;
    } else {
      update_buf->int_array.WaitToRead();
    }
  }

  void DecodeRowIds(const ps::SArray<ps::Key> &keys, int64_t *indices,
                    const int64_t master_key, const int64_t num_rows) {
    indices[0] = 0;
    for (int64_t i = 1; i <= num_rows; i++) {
      int key = DecodeKey(keys[i]);
      auto row_id = key - master_key;
      indices[i - 1] = row_id;
    }
  }

  void AccumulateRowSparseGrads(const DataHandleType type,
                                const NDArray& recved,
                                UpdateBuf* updateBuf) {
    NDArray out(kRowSparseStorage, updateBuf->merged.shape(), Context(), true,
                has_multi_precision_copy(type) ? mshadow::kFloat32 : type.dtype);
    if (has_multi_precision_copy(type)) CopyFromTo(recved, updateBuf->temp_array);
    const NDArray& to_merge = has_multi_precision_copy(type) ? updateBuf->temp_array : recved;
    // accumulate row_sparse gradients
    // TODO(haibin) override + operator for row_sparse NDArray
    // instead of calling BinaryComputeRspRsp directly
    using namespace mshadow;
    Engine::Get()->PushAsync(
    [to_merge, updateBuf, out](RunContext ctx, Engine::CallbackOnComplete on_complete) {
      op::ElemwiseBinaryOp::ComputeEx<cpu, op::mshadow_op::plus>(
      {}, {}, {to_merge, updateBuf->merged}, {kWriteTo}, {out});
      on_complete();
    }, to_merge.ctx(), {to_merge.var(), updateBuf->merged.var()}, {out.var()},
    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
    CopyFromTo(out, &(updateBuf->merged), 0);
    updateBuf->merged.WaitToRead();
  }

  void RowSparsePullResponse(const DataHandleType type,
                             const int master_key,
                             const size_t num_rows,
                             const ps::KVMeta& req_meta,
                             const ps::KVPairs<char>& req_data,
                             ps::KVServer<char>* server) {
    if (log_verbose_) LOG(INFO) << "pull: " << master_key;
    ps::KVPairs<char> response;
    if (num_rows == 0) {
      std::vector<int> lens(req_data.keys.size(), 0);
      response.keys = req_data.keys;
      response.lens.CopyFrom(lens.begin(), lens.end());
      server->Response(req_meta, response);
      return;
    }
    const NDArray& stored = store_[master_key];
    if (has_multi_precision_copy(type)) stored.WaitToRead();
    CHECK(!stored.is_none()) << "init " << master_key << " first";
    auto shape = stored.shape();
    auto unit_len = shape.ProdShape(1, shape.ndim());
    const int num_bytes = mshadow::mshadow_sizeof(type.dtype);
    const int unit_size = unit_len * num_bytes;
    const char* data = static_cast<char *> (stored.data().dptr_);
    auto len = num_rows * unit_size;
    // concat values
    response.vals.resize(len);
    #pragma omp parallel for
    for (size_t i = 1; i <= num_rows; i++) {
      int key = DecodeKey(req_data.keys[i]);
      int64_t row_id = key - master_key;
      const auto src = data + row_id * unit_size;
      auto begin = (i - 1) * unit_size;
      auto end = i * unit_size;
      response.vals.segment(begin, end).CopyFrom(src, unit_size);
    }
    // setup response
    response.keys = req_data.keys;
    std::vector<int> lens(req_data.keys.size(), unit_len);
    lens[0] = 0;
    response.lens.CopyFrom(lens.begin(), lens.end());
    server->Response(req_meta, response);
  }

  void InitRowSparseStored(const DataHandleType type,
                           const int master_key,
                           const size_t num_rows,
                           const ps::KVMeta& req_meta,
                           const ps::KVPairs<char>& req_data,
                           ps::KVServer<char>* server) {
    auto& stored = has_multi_precision_copy(type) ? store_realt_[master_key] : store_[master_key];
    int dtype = type.dtype;
    int num_bytes = mshadow::mshadow_sizeof(dtype);
    auto unit_len = req_data.lens[1] / num_bytes;
    CHECK_GT(unit_len, 0);
    size_t ds[] = {num_rows, (size_t) unit_len};
    TShape dshape(ds, ds + 2);
    CHECK_EQ(req_data.vals.size(), num_rows * unit_len * num_bytes);
    TBlob recv_blob;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      recv_blob = TBlob(reinterpret_cast<DType*>(req_data.vals.data()), dshape, cpu::kDevMask);
    })
    NDArray recved = NDArray(recv_blob, 0);
    stored = NDArray(kRowSparseStorage, dshape, Context(), true,
                     has_multi_precision_copy(type) ? mshadow::kFloat32 : type.dtype);
    if (has_multi_precision_copy(type)) {
      store_[master_key] = NDArray(kRowSparseStorage, dshape, Context(), true, type.dtype);
    }
    Engine::Get()->PushAsync(
    [this, recved, stored, type](RunContext ctx, Engine::CallbackOnComplete on_complete) {
      NDArray rsp = stored;
      stored.CheckAndAlloc({mshadow::Shape1(recved.shape()[0])});
      mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
      using namespace mxnet::op;
      nnvm::dim_t nnr = rsp.shape()[0];
      MSHADOW_IDX_TYPE_SWITCH(rsp.aux_type(rowsparse::kIdx), IType, {
        IType* idx = rsp.aux_data(rowsparse::kIdx).dptr<IType>();
        mxnet_op::Kernel<PopulateFullIdxRspKernel, cpu>::Launch(s, nnr, idx);
      });
      TBlob rsp_data = rsp.data();
      // copies or casts as appropriate
      ndarray::Copy<cpu, cpu>(recved.data(), &rsp_data, Context(), Context(), RunContext());
      on_complete();
    }, recved.ctx(), {recved.var()}, {stored.var()},
    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
    if (has_multi_precision_copy(type)) {
      CopyFromTo(stored, store_[master_key]);
      store_[master_key].WaitToRead();
    }
    stored.WaitToRead();
    server->Response(req_meta);
  }

  void DataHandleRowSparse(const DataHandleType type, const ps::KVMeta& req_meta,
                           const ps::KVPairs<char>& req_data,
                           ps::KVServer<char>* server) {
    int master_key = DecodeKey(req_data.keys[0]);
    auto num_rows = req_data.keys.size() - 1;
    auto& stored = store_[master_key];
    if (req_meta.push) {
      CHECK_GT(req_data.lens.size(), 0) << "req_data.lens cannot be empty";
      CHECK_EQ(req_data.lens[0], 0);
      if (stored.is_none()) {
        if (log_verbose_) LOG(INFO) << "initial push: " << master_key;
        // initialization
        CHECK_GT(num_rows, 0) << "init with empty data is not supported";
        InitRowSparseStored(type, master_key, num_rows, req_meta, req_data, server);
        return;
      } else {
        if (log_verbose_) LOG(INFO) << "push: " << master_key << " " << req_data.keys;
        auto& updates = update_buf_[master_key];
        if (sync_mode_ && updates.merged.is_none()) {
          updates.merged = NDArray(kRowSparseStorage, stored.shape(), Context(), true,
                                   has_multi_precision_copy(type) ? mshadow::kFloat32 : type.dtype);
        }
        if (has_multi_precision_copy(type) && updates.temp_array.is_none()) {
          updates.temp_array = NDArray(kRowSparseStorage, stored.shape(), Context(), false,
                                       mshadow::kFloat32);
        }

        if (num_rows == 0) {
          if (sync_mode_) {
            if (updates.request.empty()) {
              // reset to zeros
              int merged_dtype = has_multi_precision_copy(type) ? mshadow::kFloat32 : type.dtype;
              updates.merged = NDArray(kRowSparseStorage, stored.shape(), Context(),
                                       true, merged_dtype);
            }  // else nothing to aggregate
            updates.request.push_back(req_meta);
            ApplyUpdates(type, master_key, &updates, server);
          } else {
            server->Response(req_meta);
          }
        } else {
          auto unit_len = req_data.lens[1] / mshadow::mshadow_sizeof(type.dtype);
          CHECK_GT(unit_len, 0);
          // indices
          std::vector<int64_t> indices(num_rows);
          DecodeRowIds(req_data.keys, indices.data(), master_key, num_rows);

          // data
          TBlob idx_blob(indices.data(), mshadow::Shape1(num_rows), cpu::kDevMask);
          size_t ds[] = {(size_t) num_rows, (size_t) unit_len};
          TShape dshape(ds, ds + 2);
          TBlob recv_blob;
          MSHADOW_REAL_TYPE_SWITCH(type.dtype, DType, {
            recv_blob = TBlob(reinterpret_cast<DType*>(req_data.vals.data()),
                              dshape, cpu::kDevMask);
          })
          // row_sparse NDArray
          NDArray recved(kRowSparseStorage, stored.shape(), recv_blob, {idx_blob}, 0);

          if (updates.request.empty()) {
            if (sync_mode_) {
              CopyFromTo(recved, updates.merged);
            } else {
              if (has_multi_precision_copy(type)) {
                CopyFromTo(recved, updates.temp_array);
              } else {
                updates.temp_array = recved;
              }
            }
          } else {
            CHECK(sync_mode_);
            AccumulateRowSparseGrads(type, recved, &updates);
          }
          updates.request.push_back(req_meta);
          ApplyUpdates(type, master_key, &updates, server);
        }
      }
    } else {
      // pull
      RowSparsePullResponse(type, master_key, num_rows, req_meta, req_data, server);
    }
  }

  void DefaultStorageResponse(const DataHandleType type,
                              const int key,
                              const ps::KVMeta& req_meta,
                              const ps::KVPairs<char> &req_data,
                              ps::KVServer<char>* server) {
    NDArray* response_arr;
    if (type.requestType == RequestType::kCompressedPull) {
      response_arr = &update_buf_[key].requantized;
      CHECK(!response_arr->is_none()) << "rank " << ps::MyRank() << " : "
        << "Cannot handle compressed pull before a compressed push for key "<< key;
    } else if (type.requestType == RequestType::kCompressedFullPull ||
               type.requestType == RequestType::kDefaultPushPull) {
      // as server returns when store_realt is ready in this case
      if (has_multi_precision_copy(type)) store_[key].WaitToRead();
      response_arr = &store_[key];
      CHECK(!response_arr->is_none()) << "rank " << ps::MyRank() << " : "
        << "Cannot handle pull before a push. Init key " << key << " first";
    } else {
      LOG(FATAL) << "Unexpected pull command to server "<<static_cast<int>(type.requestType);
    }
    ps::KVPairs<char> response;
    auto len = response_arr->shape().Size() * mshadow::mshadow_sizeof(response_arr->dtype());
    response.keys = req_data.keys;
    response.lens = {len};
    // TODO(mli) try to remove this CopyFrom
    response.vals.CopyFrom(static_cast<const char*>(response_arr->data().dptr_), len);
    server->Response(req_meta, response);
  }

  void DataHandleCompressed(const DataHandleType type,
                            const ps::KVMeta& req_meta,
                            const ps::KVPairs<char> &req_data,
                            ps::KVServer<char>* server) {
    CHECK_EQ(type.dtype, mshadow::kFloat32)
      << "Gradient compression is currently supported for fp32 only";
    if (req_meta.push) {
      // there used several WaitToRead, this is because \a recved's memory
      // could be deallocated when this function returns. so we need to make sure
      // the operators with \a NDArray are actually finished

      // first for dummy key which represents original size of array, whose len is 0
      CHECK_EQ(req_data.keys.size(), (size_t)2);
      CHECK_EQ(req_data.lens.size(), (size_t)2);
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[1]);

      int original_size = DecodeKey(req_data.keys[0]);
      int key = DecodeKey(req_data.keys[1]);

      size_t ds[] = {(size_t)req_data.lens[1] / mshadow::mshadow_sizeof(type.dtype)};
      TShape dshape(ds, ds + 1);
      TBlob recv_blob(reinterpret_cast<real_t*>(req_data.vals.data()), dshape, cpu::kDevMask);
      NDArray recved = NDArray(recv_blob, 0);

      auto& stored = store_[key];
      dshape = TShape{(int64_t) original_size};

//      && gradient_compression_->get_server_compression_type() == CompressionType::kNone) {
//        // init
//        stored = NDArray(dshape, Context());
//        gradient_compression_->Dequantize(recved, &stored, 0);
//        server->Response(req_meta);
//        stored.WaitToRead();
//      } else
      if (sync_mode_) {
        CHECK(!stored.is_none()) << "Init of a key has to be uncompressed";
        // synced push
        auto& updates = update_buf_[key];
        if (gradient_compression_->get_server_compression_type() == CompressionType::kNone) {
          if (updates.merged.is_none()) {
            updates.merged = NDArray(dshape, Context());
            updates.merged = 0;
          }
          //adds to merged
          gradient_compression_->Dequantize(recved, &updates.merged, 0);
          updates.request.push_back(req_meta);
          ApplyUpdates(type, key, &updates, server, true);
        } else {
          if (updates.int_array.is_none()) {
            updates.int_array = NDArray(dshape, Context(), false, mshadow::kInt32);
            updates.int_array = 0;
            TShape recompressed_shape = TShape{gradient_compression_->
            GetServerRecompressedSize((int64_t) original_size)};
            updates.requantized = NDArray(recompressed_shape, Context());
          }
          gradient_compression_->DequantizeForSum(recved, &updates.int_array, 0);
          updates.request.push_back(req_meta);
          RecompressUpdates(key, &updates, server, original_size);
        }
      } else {
        CHECK(!stored.is_none()) << "TODO";
        CHECK(gradient_compression_->get_server_compression_type() == CompressionType::kNone)
          << "Gradient compression for async mode with server recompression is not supported";
        auto &updates = update_buf_[key];
        if (updates.temp_array.is_none()) {
          updates.temp_array = NDArray(dshape, Context());
          updates.temp_array = 0;
        }
        gradient_compression_->Dequantize(recved, &updates.temp_array, 0);
        exec_.Exec([this, key, &updates, &stored]() {
          CHECK(updater_) << "Updater is required to be set on kvstore server when async mode is used";
          updater_(key, updates.temp_array, &stored);
        });
        server->Response(req_meta);
        stored.WaitToRead();
      }
    } else {       // pull
      CHECK_EQ(req_data.keys.size(), (size_t)1);
      CHECK_EQ(req_data.lens.size(), (size_t)0);
      int key = DecodeKey(req_data.keys[0]);
      DefaultStorageResponse(type, key, req_meta, req_data, server);
    }
  }

  void DataHandleDefault(const DataHandleType type, const ps::KVMeta& req_meta,
                         const ps::KVPairs<char> &req_data,
                         ps::KVServer<char>* server) {
    // do some check
    CHECK_EQ(req_data.keys.size(), (size_t)1);
    if (req_meta.push) {
      CHECK_EQ(req_data.lens.size(), (size_t)1);
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
    }
    int key = DecodeKey(req_data.keys[0]);
    auto& stored = has_multi_precision_copy(type) ? store_realt_[key] : store_[key];
    // there used several WaitToRead, this is because \a recved's memory
    // could be deallocated when this function returns. so we need to make sure
    // the operators with \a NDArray are actually finished
    if (req_meta.push) {
      LOG(INFO) << "received push";
      size_t ds[] = {(size_t) req_data.lens[0] / mshadow::mshadow_sizeof(type.dtype)};
      TShape dshape(ds, ds + 1);
      TBlob recv_blob;
      MSHADOW_REAL_TYPE_SWITCH(type.dtype, DType, {
        recv_blob = TBlob(reinterpret_cast<DType*>(req_data.vals.data()), dshape, cpu::kDevMask);
      })
      NDArray recved = NDArray(recv_blob, 0);
      if (stored.is_none()) {
        // initialization
        stored = NDArray(dshape, Context(), false,
                         has_multi_precision_copy(type) ? mshadow::kFloat32 : type.dtype);
        CopyFromTo(recved, &stored, 0);
        server->Response(req_meta);
        if (has_multi_precision_copy(type)) {
          auto& stored_dtype = store_[key];
          stored_dtype = NDArray(dshape, Context(), false, type.dtype);
          CopyFromTo(stored, stored_dtype);
          stored_dtype.WaitToRead();
        }
        stored.WaitToRead();
        LOG(INFO) << "rank: " << ps::MyRank() << " inited key "<< key;
      } else {
        auto &updates = update_buf_[key];
        if (sync_mode_ && updates.merged.is_none()) {
          updates.merged = NDArray(dshape, Context(), false,
                                   has_multi_precision_copy(type) ? mshadow::kFloat32 : type.dtype);
        }
        if (has_multi_precision_copy(type) && updates.temp_array.is_none()) {
          updates.temp_array = NDArray(dshape, Context(), false, mshadow::kFloat32);
        }
        if (updates.request.empty()) {
          if (sync_mode_) {
            CopyFromTo(recved, updates.merged);
          } else {
            if (has_multi_precision_copy(type)) {
              CopyFromTo(recved, updates.temp_array);
            } else {
              updates.temp_array = recved;
            }
          }
        } else {
          CHECK(sync_mode_);
          if (has_multi_precision_copy(type)) {
            CopyFromTo(recved, updates.temp_array);
            updates.merged += updates.temp_array;
          } else {
            updates.merged += recved;
          }
        }
        updates.request.push_back(req_meta);
        ApplyUpdates(type, key, &updates, server);
      }
    } else {
      DefaultStorageResponse(type, key, req_meta, req_data, server);
    }
  }

  int DecodeKey(ps::Key key) {
    auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
    return key - kr.begin();
  }


  /**
   * \brief user defined mode for push
   */
  bool sync_mode_;
  KVStore::Controller controller_;
  KVStore::Updater updater_;

  /**
   * \brief store_ contains the value at kvstore for each key
   */
  std::unordered_map<int, NDArray> store_;
  std::unordered_map<int, NDArray> store_realt_;

  /**
   * \brief merge_buf_ is a buffer used if sync_mode is true. It represents
   * values from different workers being merged. The store will be updated
   * to this value when values from all workers are pushed into this buffer.
   */
  std::unordered_map<int, UpdateBuf> update_buf_;

  Executor exec_;
  ps::KVServer<char>* ps_server_;

  // whether to LOG verbose information
  bool log_verbose_;

  /*
   * \brief whether to use multi precision mode.
   * in multi precision mode, all weights are stored as float32.
   * any gradient received will be cast to float32 before accumulation and updating of weights.
   */
  bool multi_precision_;

  /**
   * \brief gradient compression object.
   * starts with none, used after SetGradientCompression sets the type
   * currently there is no support for unsetting gradient compression
   */
  std::shared_ptr<kvstore::GradientCompression> gradient_compression_;
};

}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
