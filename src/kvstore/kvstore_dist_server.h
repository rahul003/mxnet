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
#include <ps/ps.h>
#include <mxnet/kvstore.h>
#include "../operator/tensor/elemwise_binary_op-inl.h"
#include "../operator/tensor/init_op.h"
#include "gradient_compression.h"

namespace mxnet {
namespace kvstore {

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

enum class CommandType {
  kController, kStopServer, kSyncMode, kSetGradientCompression
};

enum class DataHandleType {
  kDefaultPushPull, kCompressedPush, kCompressedPull, kCompressedFullPull, kRowSparsePushPull
};

class KVStoreDistServer {
 public:
  KVStoreDistServer() {
    using namespace std::placeholders;
    ps_server_ = new ps::KVServer<float>(0);
    static_cast<ps::SimpleApp*>(ps_server_)->set_request_handle(
        std::bind(&KVStoreDistServer::CommandHandle, this, _1, _2));
    ps_server_->set_request_handle(
        std::bind(&KVStoreDistServer::DataHandleEx, this, _1, _2, _3));
    sync_mode_ = false;
    gradient_compression_ = std::make_shared<GradientCompression>();
    log_verbose_row_sparse_ = dmlc::GetEnv("MXNET_KVSTORE_DIST_ROW_SPARSE_VERBOSE", false);
    log_verbose_gc_ = dmlc::GetEnv("MXNET_KVSTORE_DIST_COMPRESSION_VERBOSE", false);
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
  struct MergeBuf {
    std::vector<ps::KVMeta> request;
    NDArray array;

    NDArray int_array;
    // used for gradient compression to merge quantized array
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
      gradient_compression_->SetNumWorkers(ps::NumWorkers());
    } else {
      // this uses value 0 for message id from frontend
      // let the main thread to execute ctrl, which is necessary for python
      exec_.Exec([this, recved]() {
          CHECK(controller_);
          controller_(recved.head, recved.body);
        });
    }
    app->Response(recved);
  }

  void DataHandleEx(const ps::KVMeta& req_meta,
                    const ps::KVPairs<real_t>& req_data,
                    ps::KVServer<real_t>* server) {
    DataHandleType recved_type = static_cast<DataHandleType>(req_meta.cmd);
    if (recved_type == DataHandleType::kRowSparsePushPull) {
      DataHandleRowSparse(req_meta, req_data, server);
    } else if (recved_type == DataHandleType::kCompressedPush ||
               recved_type == DataHandleType::kCompressedFullPull ||
               recved_type == DataHandleType::kCompressedPull ) {
      DataHandleCompressed(req_meta, req_data, server);
    } else {
      DataHandleDefault(req_meta, req_data, server);
    }
    return;
  }

  inline void ApplyUpdates(const int key, MergeBuf *merged,
                           NDArray *stored,
                           ps::KVServer<real_t>* server,
                           bool reset_merged = false) {
    if (merged->request.size() == (size_t) ps::NumWorkers()) {
      // let the main thread to execute updater_, which is necessary for python
      if (updater_) {
        exec_.Exec([this, key, merged, stored](){
          CHECK(updater_);
          updater_(key, merged->array, stored);
        });
      } else {
        // if no updater, just copy
        CopyFromTo(merged->array, stored);
      }
      if (log_verbose_row_sparse_ || log_verbose_gc_)  {
        LOG(INFO) << "sync response to " << merged->request.size() << " workers";
      }
      for (const auto& req : merged->request) {
        server->Response(req);
      }
      merged->request.clear();
      stored->WaitToRead();

//      std::cout << stored->shape().Size() << " is stored size"<< std::endl;
//      std::cout << merged->array.shape().Size() << " is merged array size"<< std::endl;
//      for(int i=stored->shape().Size() - 1; i > stored->shape().Size() - 50 ; i--) {
//        std::cout << *( stored->data().dptr<float>() + i ) << " ";
//      }
//      std::cout << std::endl;

      if (reset_merged) {
        // reset needed for gradient compression
        merged->array = 0;
      }
    } else {
      merged->array.WaitToRead();
    }
  }

  inline void RecompressUpdates(const int key, MergeBuf *merged, ps::KVServer<real_t>* server, int original_size) {
    CHECK(gradient_compression_->get_server_compression_type() != CompressionType::kNone);
    if (merged->request.size() == (size_t) ps::NumWorkers()) {
      gradient_compression_->Requantize(merged->int_array, &(merged->requantized), 0);
      for (const auto &req : merged->request) {
        server->Response(req);
      }
      merged->request.clear();
      merged->requantized.WaitToRead();
      // reset int array for gradient compression
      merged->int_array = 0;
    } else {
      merged->int_array.WaitToRead();
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

  void DataHandleRowSparse(const ps::KVMeta& req_meta,
                       const ps::KVPairs<real_t>& req_data,
                       ps::KVServer<real_t>* server) {
    int master_key = DecodeKey(req_data.keys[0]);
    auto num_rows = req_data.keys.size() - 1;
    auto& stored = store_[master_key];
    if (req_meta.push) {
      CHECK_GT(req_data.lens.size(), 0) << "req_data.lens cannot be empty";
      CHECK_EQ(req_data.lens[0], 0);
      real_t* data = req_data.vals.data();
      if (stored.is_none()) {
        if (log_verbose_row_sparse_) LOG(INFO) << "initial push: " << master_key;
        // initialization
        CHECK_GT(num_rows, 0) << "init with empty data is not supported";
        auto unit_len = req_data.lens[1];
        CHECK_GT(unit_len, 0);
        size_t ds[] = {num_rows, (size_t) unit_len};
        TShape dshape(ds, ds + 2);
        CHECK_EQ(req_data.vals.size(), num_rows * unit_len);
        TBlob recv_blob(data, dshape, cpu::kDevMask);  // NOLINT(*)
        NDArray recved = NDArray(recv_blob, 0);
        stored = NDArray(kRowSparseStorage, dshape, Context());
        Engine::Get()->PushAsync(
          [recved, stored](RunContext ctx, Engine::CallbackOnComplete on_complete) {
            NDArray rsp = stored;
            stored.CheckAndAlloc({mshadow::Shape1(recved.shape()[0])});
            mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
            using namespace mxnet::op;
            nnvm::dim_t nnr = rsp.shape()[0];
            MSHADOW_IDX_TYPE_SWITCH(rsp.aux_type(rowsparse::kIdx), IType, {
              IType* idx = rsp.aux_data(rowsparse::kIdx).dptr<IType>();
              mxnet_op::Kernel<PopulateFullIdxRspKernel, cpu>::Launch(s, nnr, idx);
            });
            mshadow::Copy(rsp.data().FlatTo1D<cpu, float>(),
                          recved.data().FlatTo1D<cpu, float>(), s);
            on_complete();
          }, recved.ctx(), {recved.var()}, {stored.var()},
          FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
        stored.WaitToRead();
        server->Response(req_meta);
        return;
      }
      // synced push
      if (sync_mode_) {
        if (log_verbose_row_sparse_) LOG(INFO) << "sync push: " << master_key << " " << req_data.keys;
        auto& merged = merge_buf_[master_key];
        if (merged.array.is_none()) {
          merged.array = NDArray(kRowSparseStorage, stored.shape(), Context());
        }
        if (num_rows == 0) {
          // reset to zeros
          if (merged.request.size() == 0) {
            merged.array = NDArray(kRowSparseStorage, stored.shape(), Context());
          } else {
            // nothing to aggregate
          }
          merged.request.push_back(req_meta);
          ApplyUpdates(master_key, &merged,  &stored, server);
          return;
        }
        auto unit_len = req_data.lens[1];
        CHECK_GT(unit_len, 0);
        // indices
        std::vector<int64_t> indices(num_rows);
        DecodeRowIds(req_data.keys, indices.data(), master_key, num_rows);
        // data
        TBlob idx_blob(indices.data(), mshadow::Shape1(num_rows), cpu::kDevMask);
        size_t ds[] = {(size_t) num_rows, (size_t) unit_len};
        TShape dshape(ds, ds + 2);
        TBlob recv_blob(data, dshape, cpu::kDevMask); // NOLINT(*)
        // row_sparse NDArray
        NDArray recved(kRowSparseStorage, stored.shape(), recv_blob, {idx_blob}, 0);

        if (merged.request.size() == 0) {
          CopyFromTo(recved, &merged.array, 0);
        } else {
          NDArray out(kRowSparseStorage, stored.shape(), Context());
          std::vector<Engine::VarHandle> const_vars;
          const_vars.push_back(recved.var());
          const_vars.push_back(merged.array.var());
          // accumulate row_sparse gradients
          // TODO(haibin) override + operator for row_sparse NDArray
          // instead of calling BinaryComputeRspRsp directly
          using namespace mshadow;
          Engine::Get()->PushAsync(
            [recved, merged, out](RunContext ctx, Engine::CallbackOnComplete on_complete) {
              op::ElemwiseBinaryOp::ComputeEx<cpu, op::mshadow_op::plus>(
                {}, {}, {recved, merged.array}, {kWriteTo}, {out});
              on_complete();
            }, recved.ctx(), const_vars, {out.var()},
            FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
          CopyFromTo(out, &merged.array, 0);
        }
        merged.request.push_back(req_meta);
        ApplyUpdates(master_key, &merged,  &stored, server);
      } else {
        // async push
        if (log_verbose_row_sparse_) LOG(INFO) << "async push: " << master_key;
        if (num_rows == 0) {
          server->Response(req_meta);
          return;
        }
        auto unit_len = req_data.lens[1];
        CHECK_GT(unit_len, 0);
        // indices
        std::vector<int64_t> indices(num_rows);
        DecodeRowIds(req_data.keys, indices.data(), master_key, num_rows);
        TBlob idx_blob(indices.data(), mshadow::Shape1(num_rows), cpu::kDevMask);
        size_t ds[] = {(size_t) num_rows, (size_t) unit_len};
        TShape dshape(ds, ds + 2);
        TBlob recv_blob(data, dshape, cpu::kDevMask); // NOLINT(*)
        NDArray recved(kRowSparseStorage, stored.shape(), recv_blob, {idx_blob}, 0);
        exec_.Exec([this, master_key, &recved, &stored](){
            CHECK(updater_);
            updater_(master_key, recved, &stored);
          });
        server->Response(req_meta);
        stored.WaitToRead();
      }
    } else {
      // pull
      if (log_verbose_row_sparse_) LOG(INFO) << "pull: " << master_key;
      ps::KVPairs<real_t> response;
      if (num_rows == 0) {
        std::vector<int> lens(req_data.keys.size(), 0);
        response.keys = req_data.keys;
        response.lens.CopyFrom(lens.begin(), lens.end());
        server->Response(req_meta, response);
        return;
      }
      CHECK(!stored.is_none()) << "init " << master_key << " first";
      auto shape = stored.shape();
      auto unit_len = shape.ProdShape(1, shape.ndim());
      const float* data = stored.data().dptr<float>();
      auto len = unit_len * num_rows;
      // concat values
      response.vals.resize(len);
      #pragma omp parallel for
      for (size_t i = 1; i <= num_rows; i++) {
        int key = DecodeKey(req_data.keys[i]);
        int64_t row_id = key - master_key;
        const auto src = data + row_id * unit_len;
        auto begin = (i - 1) * unit_len;
        auto end = i * unit_len;
        response.vals.segment(begin, end).CopyFrom(src, unit_len);
      }
      // setup response
      response.keys = req_data.keys;
      std::vector<int> lens(req_data.keys.size(), unit_len);
      lens[0] = 0;
      response.lens.CopyFrom(lens.begin(), lens.end());
      server->Response(req_meta, response);
    }
  }

  void DefaultStorageResponse(int key,
                              const ps::KVMeta& req_meta,
                              const ps::KVPairs<real_t> &req_data,
                              ps::KVServer<real_t>* server,
                              DataHandleType mode) {
    NDArray* response_arr;
    if (mode == DataHandleType::kCompressedPull) {
      response_arr = &merge_buf_[key].requantized;
      CHECK(!response_arr->is_none()) << " rank " << ps::MyRank() << " : "
        << "Cannot handle compressed pull before a compressed push"
        << " for key " << key;
    } else if (mode == DataHandleType::kCompressedFullPull ||
               mode == DataHandleType::kDefaultPushPull) {
      response_arr = &store_[key];
      CHECK(!response_arr->is_none()) << " rank " << ps::MyRank() << " : "
        << "Cannot handle pull before a push. "
        << " Init key " << key << " first";
    } else {
      LOG(FATAL) << "Unexpected command to server " << req_meta.cmd;
    }

   ps::KVPairs<real_t> response;
   auto len = response_arr->shape().Size();
   response.keys = req_data.keys;
   response.lens = {len};
   // TODO(mli) try to remove this CopyFrom
   response.vals.CopyFrom(static_cast<const float*>(response_arr->data().dptr_), len);
   server->Response(req_meta, response);
  }

  void DataHandleCompressed(const ps::KVMeta& req_meta,
                            const ps::KVPairs<real_t> &req_data,
                            ps::KVServer<real_t>* server) {

    if (log_verbose_gc_) {
      LOG(INFO) << "rank " << ps::MyRank() << ": Received request for " << req_meta.cmd
                << (req_meta.push ? "compressed push" :
                     ( static_cast<DataHandleType>(req_meta.cmd) == DataHandleType::kCompressedFullPull ?
                        "full pull" : "compressed pull"));
    }

    if (req_meta.push) {
      // first for dummy key which represents original size of array, whose len is 0
      CHECK_EQ(req_data.keys.size(), (size_t)2);
      CHECK_EQ(req_data.lens.size(), (size_t)2);
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[1]);

      int original_size = DecodeKey(req_data.keys[0]);
      int key = DecodeKey(req_data.keys[1]);
      std::cout << "original size in server " << original_size << std::endl;
      size_t ds[] = {(size_t)req_data.lens[1]};
      TShape dshape(ds, ds + 1);
      TBlob recv_blob((real_t*) req_data.vals.data(), dshape, cpu::kDevMask);
      NDArray recved = NDArray(recv_blob, 0);
      dshape = TShape{(int64_t) original_size};

      auto& stored = store_[key];
      if (stored.is_none() && gradient_compression_->get_server_compression_type() == CompressionType::kNone) {
        // initialization
        stored = NDArray(dshape, Context());
      }

      if (sync_mode_) {
        // synced push
        auto& merged = merge_buf_[key];
        if (gradient_compression_->get_server_compression_type() == CompressionType::kNone) {
          CHECK(!stored.is_none()) << "stored is none for key "<<key;
          if (merged.array.is_none()) {
            merged.array = NDArray(dshape, Context());
            merged.array = 0;
          }
          gradient_compression_->Dequantize(recved, &merged.array, 0);
          merged.request.push_back(req_meta);
          ApplyUpdates(key, &merged, &stored, server, true);
        } else {
          if (merged.int_array.is_none()) {
            merged.int_array = NDArray(dshape, Context(), false, mshadow::kInt32);
            merged.int_array = 0;
            TShape recompressed_shape = TShape{gradient_compression_->
            GetServerRecompressedSize((int64_t) original_size)};
            merged.requantized = NDArray(recompressed_shape, Context());
          }
          gradient_compression_->DequantizeForSum(recved, &merged.int_array, 0);
          merged.request.push_back(req_meta);
          RecompressUpdates(key, &merged, server, original_size);
        }
      } else {
        // async push
        CHECK(gradient_compression_->get_server_compression_type() == CompressionType::kNone)
          << "Gradient compression for async mode with server recompression is not supported";
        auto &merged = merge_buf_[key];
        if (merged.array.is_none()) {
          merged.array = NDArray(dshape, Context());
          merged.array = 0;
        }
        gradient_compression_->Dequantize(recved, &merged.array, 0);
        exec_.Exec([this, key, &merged, &stored]() {
          CHECK(updater_) << "Updater is required to be set on kvstore server when async mode is used";
          updater_(key, merged.array, &stored);
        });
        server->Response(req_meta);
        stored.WaitToRead();
      }
    } else {       // pull
      CHECK_EQ(req_data.keys.size(), (size_t)1);
      CHECK_EQ(req_data.lens.size(), (size_t)0);
      int key = DecodeKey(req_data.keys[0]);
      DataHandleType recved_type = static_cast<DataHandleType>(req_meta.cmd);
      DefaultStorageResponse(key, req_meta, req_data, server, recved_type);
    }
  }



  void DataHandleDefault(const ps::KVMeta& req_meta,
                         const ps::KVPairs<real_t> &req_data,
                         ps::KVServer<real_t>* server) {
    CHECK_EQ(req_meta.cmd, static_cast<int>(DataHandleType::kDefaultPushPull));
    // do some check
    CHECK_EQ(req_data.keys.size(), (size_t)1);
    if (req_meta.push) {
      CHECK_EQ(req_data.lens.size(), (size_t)1);
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
    }

    int key = DecodeKey(req_data.keys[0]);
    auto& stored = store_[key];

    if (log_verbose_gc_) {
      LOG(INFO) << "rank " << ps::MyRank() << ": Received default request for "
                << (req_meta.push ? "push" : "pull") << " for key " << key;
    }

    // there used several WaitToRead, this is because \a recved's memory
    // could be deallocated when this function returns. so we need to make sure
    // the operators with \a NDArray are actually finished
    if (req_meta.push) {
      size_t ds[] = {(size_t)req_data.lens[0]};
      TShape dshape(ds, ds + 1);
      TBlob recv_blob((real_t*)req_data.vals.data(), // NOLINT(*)
                      dshape, cpu::kDevMask);
      NDArray recved = NDArray(recv_blob, 0);
      if (stored.is_none()) {
        // initialization
        stored = NDArray(dshape, Context());
        CopyFromTo(recved, &stored, 0);
        server->Response(req_meta);
        stored.WaitToRead();
      } else if (sync_mode_) {
        // synced push
        auto& merged = merge_buf_[key];
        if (merged.array.is_none()) {
          merged.array = NDArray(dshape, Context());
        }
        if (merged.request.size() == 0) {
          CopyFromTo(recved, &merged.array, 0);
        } else {
          merged.array += recved;
        }
        merged.request.push_back(req_meta);
        ApplyUpdates(key, &merged, &stored, server);
      } else {
        // async push
        exec_.Exec([this, key, &recved, &stored](){
            CHECK(updater_);
            updater_(key, recved, &stored);
          });
        server->Response(req_meta);
        stored.WaitToRead();
      }
    } else {
      DefaultStorageResponse(key, req_meta, req_data, server,
                             static_cast<DataHandleType>(req_meta.cmd));
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

  /**
   * \brief merge_buf_ is a buffer used if sync_mode is true. It represents
   * values from different workers being merged. The store will be updated
   * to this value when values from all workers are pushed into this buffer.
   */
  std::unordered_map<int, MergeBuf> merge_buf_;

  Executor exec_;
  ps::KVServer<float>* ps_server_;

  // whether to LOG verbose information
  bool log_verbose_row_sparse_;
  bool log_verbose_gc_;

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
