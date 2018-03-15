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

/**
 * Copyright (c) 2015 by Contributors
 * @file   kvstore_dist.h
 * @brief  distributed implementation based on ps-lite
 */
#ifndef MXNET_KVSTORE_KVSTORE_DIST_H_
#define MXNET_KVSTORE_KVSTORE_DIST_H_
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include "./kvstore_local.h"
#include "mxnet/engine.h"
#include "ps/ps.h"
#include "./kvstore_dist_server.h"

#if MKL_EXPERIMENTAL == 1
#include <mkl_memory.h>
#include "../operator/mkl/mkl_memory-inl.h"
#include "../operator/mkl/mkl_util-inl.h"
#endif
namespace mxnet {
namespace kvstore {

/**
 * \brief distributed kvstore
 *
 * it's the server node's job to control the data consistency among all
 * workers. see details on \ref ServerHandle::Start
 */
class KVStoreDist : public KVStoreLocal {
 public:
  explicit KVStoreDist(bool use_device_comm)
      : KVStoreLocal(use_device_comm), ps_worker_(nullptr), server_(nullptr) {
    if (IsWorkerNode()) {
      int new_customer_id = GetNewCustomerId();
      ps_worker_ = new ps::KVWorker<real_t>(0, new_customer_id);
      ps::StartAsync(new_customer_id, "mxnet\0");
      if (!ps::Postoffice::Get()->is_recovery()) {
        ps::Postoffice::Get()->Barrier(
          new_customer_id,
          ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
      }
    }
    bigarray_bound_ = dmlc::GetEnv("MXNET_KVSTORE_BIGARRAY_BOUND", 1000 * 1000);
    log_verbose_row_sparse_ = dmlc::GetEnv("MXNET_KVSTORE_DIST_ROW_SPARSE_VERBOSE", false);
    log_verbose_gc_ = dmlc::GetEnv("MXNET_KVSTORE_DIST_COMPRESSION_VERBOSE", false);
  }

  virtual ~KVStoreDist() {
    Engine::Get()->WaitForAll();
    if (IsWorkerNode()) {
      if (barrier_before_exit_) {
        Barrier();
        if (get_rank() == 0 && ps_worker_->get_customer()->customer_id() == 0) {
          // stop the executor at servers
          SendCommandToServers(static_cast<int>(CommandType::kStopServer), "");
        }
      }
      ps::Finalize(ps_worker_->get_customer()->customer_id(), barrier_before_exit_);
      delete ps_worker_;
    }
  }

  void set_updater(const Updater& updater) override {
    CHECK(updater) << "invalid updater";
    if (IsServerNode()) {
      CHECK_NOTNULL(server_)->set_updater(updater);
    } else {
      updater_ = updater;
    }
  }

  void SetGradientCompression(const std::vector<std::pair<std::string, std::string> >
                              & kwargs) override {
    gradient_compression_->SetParams(kwargs);
    gradient_compression_->SetNumWorkers(ps::NumWorkers());
    if (log_verbose_gc_) {
      LOG(INFO) << "rank " << ps::MyRank() << ": Setting gradient compression of type "
                << gradient_compression_->get_type_str() << " with server compression type "
                << gradient_compression_->get_server_compression_type_str() <<
                " and compression step of " << gradient_compression_->get_compression_step_str();
    }
    if (get_rank() == 0) {
      SendCommandToServers(static_cast<int>(CommandType::kSetGradientCompression),
                           gradient_compression_->EncodeParams());
    }
  }

  void Barrier() override {
    ps::Postoffice::Get()->Barrier(ps_worker_->get_customer()->customer_id(), ps::kWorkerGroup);
  }

  void SendCommandToServers(int cmd_id,
                            const std::string& cmd_body) override {
    CHECK_NOTNULL(ps_worker_);
    ps_worker_->Wait(ps_worker_->Request(cmd_id, cmd_body, ps::kServerGroup));
  }

  void SetSyncMode() override {
    if (get_rank() == 0) {
      SendCommandToServers(static_cast<int>(kvstore::CommandType::kSyncMode), "");
    }
    sync_mode_  = true;
  }

  int get_group_size() const override { return ps::NumWorkers(); }

  int get_rank() const override { return ps::MyRank(); }

  int get_num_dead_node(int node_id, int timeout) const override {
    int number = 0;
    auto dead_nodes = ps::Postoffice::Get()->GetDeadNodes(timeout);
    const auto& watch_nodes = ps::Postoffice::Get()->GetNodeIDs(node_id);
    std::unordered_set<int> watch_set(watch_nodes.begin(), watch_nodes.end());
    for (int r : dead_nodes) {
      if (watch_set.find(r) != watch_set.end()) number++;
    }
    return number;
  }

  void RunServer(const Controller& controller) override {
    CHECK(!IsWorkerNode());
    if (IsServerNode()) {
      server_ = new KVStoreDistServer();
      server_->set_controller(controller);
    }

    ps::StartAsync(0, "mxnet_server\0");
    if (!ps::Postoffice::Get()->is_recovery()) {
      ps::Postoffice::Get()->Barrier(0,
        ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
    }
    if (server_) server_->Run();
    // if there's a server started then worker should run this to receive compressed gradients

    ps::Finalize(0, true);
    if (server_) {
      delete server_;
    }
    server_ = nullptr;
  }

 private:
  static std::atomic<int> customer_id_;

  static int GetNewCustomerId() {
    return customer_id_++;
  }


  /**
   * \brief struct for ps keys and lens
   */
  struct PSKV {
    ps::SArray<ps::Key> keys;  // n keys
    ps::SArray<int> lens;  // the length of the i-th value
    int size;
  };

  struct ComprPSKV {
    PSKV push;
    PSKV pull;
    PSKV full_pull;
  };

  /**
   * \brief cache all key partitions
   *
   * `ps_kv_` is used for pushes and pulls without gradient compression.
   *
   * `compr_ps_kv_` is used for gradient compression. It contains different
   * pskvs for push, pull and full_pull.
   * Push and pull represent compressed push and compressed pull according to type of gradient compression.
   * Full_pull is used in certain cases like initialization or when server does not recompress parameters.
   * Note: `compr_ps_kv_.full_pull` for some key k may not be the same as `ps_kv_[k].pull`.
   * This is because sharding may cause slightly different divisions when size is
   * not perfectly divisible.
   */
  std::unordered_map<int, PSKV> ps_kv_;
  std::unordered_map<int, ComprPSKV> compr_ps_kv_;

  /**
   * \brief serialize access to ps_kv_ or push_ps_kv_/pull_ps_kv_ while encoding keys
   */
  std::mutex mu_;

  void InitImpl(const std::vector<int>& keys,
                const std::vector<NDArray>& values) override {
    CheckUnique(keys);
    for (size_t i = 0; i < keys.size(); ++i) {
      comm_->Init(keys[i], values[i].storage_type(), values[i].shape(), values[i].dtype());
    }
    if (get_rank() == 0) {
      Push_(keys, values, 0, false);
      // wait until the push is finished
      for (const int key : keys) {
        send_buf_[key].WaitToWrite();
        recv_buf_[key].WaitToWrite();
        send_compr_buf_[key].WaitToWrite();
      }
    } else {
      // do nothing
    }
    if (!ps::Postoffice::Get()->is_recovery()) {
      Barrier();
    }
  }

  void PushImpl(const std::vector<int>& keys,
                const std::vector<NDArray>& values,
                int priority) override {
    Push_(keys, values, priority, true);
  }

  void PullImpl(const std::vector<int>& keys,
                const std::vector<NDArray*>& values,
                int priority) override {
    if (gradient_compression_->get_type() == CompressionType::kNone) {
      PullImplDefault(keys, values, priority);
    } else {
      PullImplCompressed(keys, values, priority);
    }
  }

  void PullImplDefault(const std::vector<int>& keys,
                       const std::vector<NDArray*>& values,
                       int priority) {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray *> > grouped_vals;
    GroupKVPairsPull(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      auto &recv_buf = recv_buf_[key];
      auto &send_buf = send_buf_[key];
      const auto storage_type = grouped_vals[i][0]->storage_type();
      CHECK_EQ(storage_type, kDefaultStorage)
        << "Expected stype of value to be kDefaultStorage";
      if (recv_buf.is_none()) {
        // it may happen for the first time a no-rank-0 worker pull the weight.
        recv_buf = NDArray(grouped_vals[i][0]->shape(), pinned_ctx_,
                           true, grouped_vals[i][0]->dtype());
      }
      if (send_buf.is_none()) {
        send_buf = NDArray(grouped_vals[i][0]->shape(), pinned_ctx_, true, grouped_vals[i][0]->dtype());
      }
      auto pull_from_servers = [this, key, recv_buf, send_buf](
      RunContext rctx, Engine::CallbackOnComplete cb) {
        // convert to ps keys
        size_t size = recv_buf.shape().Size();

        PSKV &pskv = EncodeDefaultKey(key, size, false);
        real_t *data = recv_buf.data().dptr<real_t>();
        // false means not to delete data when SArray is deleted
        auto vals = new ps::SArray<real_t>(data, size, false);
        // issue pull
        int cmd = static_cast<int>(DataHandleType::kDefaultPushPull);
        CHECK_NOTNULL(ps_worker_)->ZPull(
        pskv.keys, vals, &pskv.lens, cmd, [vals, cb]() {
          delete vals;
          cb();
        });
      };
      CHECK_NOTNULL(Engine::Get())->PushAsync(
      pull_from_servers,
      pinned_ctx_,
      {},
      {recv_buf.var(), send_buf.var()},
      FnProperty::kNormal,
      priority,
      PROFILER_MESSAGE("KVStoreDistDefaultStoragePull"));
      comm_->Broadcast(key, recv_buf, grouped_vals[i], priority);
    }
  }

  void PullImplCompressed(const std::vector<int>& keys,
                          const std::vector<NDArray*>& values,
                          int priority) {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray*> > grouped_vals;
    GroupKVPairsPull(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      auto& recv_buf = recv_buf_[key];
      auto& send_buf = send_buf_[key];
      auto& recv_compr_buf = recv_compr_buf_[key];

      const auto storage_type = grouped_vals[i][0]->storage_type();
      CHECK_EQ(storage_type, kDefaultStorage)
        << "Expected stype of value to be kDefaultStorage";
      if (recv_buf.is_none()) {
        // it may happen for the first time a no-rank-0 worker pull the weight.
        recv_buf = NDArray(grouped_vals[i][0]->shape(), pinned_ctx_, true, grouped_vals[i][0]->dtype());
      }
      if (send_buf.is_none()) {
        send_buf = NDArray(grouped_vals[i][0]->shape(), pinned_ctx_, true, grouped_vals[i][0]->dtype());
      }
      if (gradient_compression_->get_server_compression_type() != CompressionType::kNone
          && recv_compr_buf.is_none()) {
        TShape recompr_shape = TShape{gradient_compression_->
        GetServerRecompressedSize((int64_t) grouped_vals[i][0]->shape().Size())};
        recv_compr_buf = NDArray(recompr_shape, pinned_ctx_, true, grouped_vals[i][0]->dtype());
      }

      bool full_pull = (gradient_compression_->get_server_compression_type() == CompressionType::kNone)
                       || !first_pull_done_[key];

      if (log_verbose_gc_) {
        LOG(INFO) << "rank "<< ps::MyRank() << ": Issuing a pull of type "
                  << (full_pull ? "FullPull" : "CompressedPull" ) << " for key " << key;
      }

      auto pull_from_servers = [this, key, send_buf, recv_buf, recv_compr_buf, full_pull](
      RunContext rctx, Engine::CallbackOnComplete cb) {
        int cmd;
        size_t size;
        if (full_pull) {
          cmd = static_cast<int>(DataHandleType::kCompressedFullPull);
          size = recv_buf.shape().Size();
        } else {
          cmd = static_cast<int>(DataHandleType::kCompressedPull);
          size = recv_compr_buf.shape().Size();
        }
        PSKV &pskv = EncodeCompressedKey(key, recv_buf.shape().Size(), false, !full_pull);

        real_t *data = (full_pull) ? recv_buf.data().dptr<real_t>()
                                   : recv_compr_buf.data().dptr<real_t>();

        // false means not to delete data when SArray is deleted
        auto vals = new ps::SArray<real_t>(data, size, false);

        // issue pull
        CHECK_NOTNULL(ps_worker_)->ZPull(
        pskv.keys, vals, &pskv.lens, cmd, [vals, cb]() { delete vals; cb(); });
      };

      std::vector<Engine::VarHandle> mutable_vars = {recv_buf.var(), send_buf.var()};
      //send_buf.var(),
      //send_buf is taken as write dep so that push doesn't go first
      if (gradient_compression_->get_server_compression_type() != CompressionType::kNone) {
        mutable_vars.push_back(recv_compr_buf.var());
      }

      CHECK_NOTNULL(Engine::Get())->PushAsync(
      pull_from_servers,
      pinned_ctx_,
      {},
      mutable_vars,
      FnProperty::kNormal,
      priority,
      PROFILER_MESSAGE("KVStoreDistDefaultStoragePullCompressed"));

      NDArray& stored = store_[key];
      // update weights now if required
      if (!full_pull) {
        CHECK(sync_mode_)
          << ": Async mode of kvstore dist does not support server recompression type for gradient compression";
        CHECK(!stored.is_none()) << ps::MyRank() << " stored is none";
        NDArray& decomp_buf = decomp_buf_[key];
        if (decomp_buf.is_none()) {
          decomp_buf = NDArray(grouped_vals[i][0]->shape(), pinned_ctx_, false, grouped_vals[i][0]->dtype());
        }
        gradient_compression_->DequantizeFinal(recv_compr_buf, &decomp_buf, priority);
        if (updater_) {
          CHECK(updater_);
          updater_(key, decomp_buf, &stored);
        } else {
          // TODO check if copy needed
          stored = decomp_buf;
        }
        comm_->Broadcast(key, stored, grouped_vals[i], priority);
        decomp_buf = 0;
      } else {
        // if full pull, then assumption is that server applies updater
        // this case initializes stored when server will recompress future updates
        if (stored.is_none()) {
          stored = NDArray(grouped_vals[i][0]->shape(), pinned_ctx_);
          CopyFromTo(recv_buf, stored, priority);
        } else {
          // if stored is not none, no need to copy as that is the case when we are doing full pulls always
          CHECK(gradient_compression_->get_server_compression_type() == CompressionType::kNone);
        }
        comm_->Broadcast(key, recv_buf, grouped_vals[i], priority);
      }
      if (!first_pull_done_[key]) first_pull_done_[key] = true;
    }
  }

  void PullRowSparseImpl(const std::vector<int>& keys,
                         const std::vector<std::pair<NDArray*, NDArray>>& val_rowids,
                         int priority = 0) override {
    std::vector<int> uniq_keys;
    std::vector<std::vector<std::pair<NDArray*, NDArray>>> grouped_val_rowids;
    GroupKVPairsPullRsp(keys, val_rowids, &uniq_keys, &grouped_val_rowids);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      // use the same array for merging to guarantee that pull always happens
      // after the previous push on this key
      auto& recv_buf = send_buf_[key];
      auto& grouped_val_rowid = grouped_val_rowids[i];
      const auto storage_type = grouped_val_rowid[0].first->storage_type();
      CHECK_EQ(storage_type, kRowSparseStorage)
               << "expected kRowSparseStorage, but got " << storage_type;
      if (recv_buf.is_none()) {
        // it may happen for the first time a no-rank-0 worker pull the weight.
        recv_buf = NDArray(storage_type, grouped_val_rowid[0].first->shape(),
                           pinned_ctx_, true, grouped_val_rowid[0].first->dtype());
      }
      auto &target_val_rowids = grouped_val_rowids[i];
      const size_t num_vals = target_val_rowids.size();
      for (size_t i = 0; i < num_vals; i++) {
        auto &row_id = target_val_rowids[i].second;
        target_val_rowids[i].second = Unique(row_id, pinned_ctx_, 0);
      }
      CHECK_EQ(num_vals, 1) << "RowSparsePull with multiple values is not supported yet";
      NDArray& indices = target_val_rowids[0].second;
      PullRowSparse_(key, recv_buf, indices, priority);
      // The recv_buf contains values pulled from remote server with unique indices.
      // Directly broadcast w/o rowids if num_vals == 1
      auto get_val = [](const std::pair<NDArray*, NDArray>& p) { return p.first; };
      std::vector<NDArray*> grouped_val(grouped_val_rowid.size());
      std::transform(grouped_val_rowid.begin(), grouped_val_rowid.end(),
                     grouped_val.begin(), get_val);
      comm_->Broadcast(key, recv_buf, grouped_val, priority);
    }
  }

  void Compress(int key, const NDArray& merged, int priority) {
    auto &compr_buf = send_compr_buf_[key];
    auto &res_buf = residual_[key];
    size_t original_size = merged.shape().Size();
    size_t compressed_size = gradient_compression_->GetCompressedSize(original_size);
    // Init the small buffer and residual_ buffer for quantize
    if (compr_buf.is_none()) {
      compr_buf = NDArray(TShape{(int64_t) compressed_size}, merged.ctx(), false, merged.dtype());
    }
    if (res_buf.is_none()) {
      res_buf = NDArray(TShape{(int64_t) original_size},
                        merged.ctx(), false, merged.dtype());
      res_buf = 0;
    }
    gradient_compression_->Quantize(merged, &compr_buf, &res_buf, priority);
  }

  void CopyToCommBuf(int key, const NDArray& merged, const NDArrayStorageType& storage_type, int priority) {
    auto &comm_buf = send_buf_[key];
    if (merged.ctx().dev_mask() == cpu::kDevMask) {
      // Start of a push doesn't guarantee that the previous pushes are completed.
      // This shouldn't affect training of networks though because training involves
      // a sequence of push, pull, then push. This imposes ordering that the
      // second push happens after the first pull, and the pull happens after first push.
      comm_buf = merged;
    } else {
      if (comm_buf.is_none()) {
        if (storage_type == kDefaultStorage) {
          comm_buf = NDArray(merged.shape(), pinned_ctx_, true, merged.dtype());
        } else {
          comm_buf = NDArray(storage_type, merged.shape(), pinned_ctx_, true, merged.dtype());
        }
      }
      CopyFromTo(merged, &comm_buf);
    }
  }

  void Push_(const std::vector<int>& keys,
             const std::vector<NDArray>& values,
             int priority,
             bool do_merge) {
    // first aggregate the values over keys
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray> > grouped_vals;
    GroupKVPairsPush(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      // merge over devices
      int key = uniq_keys[i];
      const auto& vals = grouped_vals[i];
      NDArray merged = do_merge ? comm_->Reduce(key, vals, priority) : vals[0];
      const auto storage_type = merged.storage_type();
      auto &comm_buf = send_buf_[key];

      // push to servers
      if (storage_type == kDefaultStorage) {
        if (gradient_compression_->get_type() == CompressionType::kNone) {
          CopyToCommBuf(key, merged, storage_type, priority);
          PSKV& pskv = EncodeDefaultKey(key, comm_buf.shape().Size(), true);
          PushDefault(key, comm_buf, pskv, priority);
        } else {
          // Note: gradient compression uses `do_merge` as proxy to
          // detect whether the push is initialization of a key or not.
          // is_active is false when push is initialization of key
          bool is_active = do_merge;
          PSKV &pskv = EncodeCompressedKey(key, merged.shape().Size(), true, is_active);
          // Returns push_pskv if active, else pull_pskv
          // we want inactive gc to send uncompressed gradients,
          // but sharded in the same way as later pushes would when gc becomes active
          if (!is_active) {
            CopyToCommBuf(key, merged, storage_type, priority);
            PushDefault(key, comm_buf, pskv, priority);
          } else {
            auto &compr_buf = send_compr_buf_[key];
            if (gradient_compression_->get_compression_step() == CompressionStep::kGpuAfterAggregation) {
              Compress(key, merged, priority);
            } else if (gradient_compression_->get_compression_step() == CompressionStep::kCpuAfterAggregation) {
              CopyToCommBuf(key, merged, storage_type, priority);
              Compress(key, comm_buf, priority);
            }
            PushCompressed(key, comm_buf, compr_buf, pskv, priority);
          }
        }
      } else {
        CHECK(gradient_compression_->get_type() == CompressionType::kNone)
          << "Gradient compression for row sparse storage type is not supported";
        if (storage_type == kRowSparseStorage) {
          PushRowSparse(key, comm_buf, priority);
        } else {
          LOG(FATAL) << "unknown storage type";
        }
      }
    }
  }

  void PushCompressed(int key, const NDArray& comm_buf, const NDArray& small_buf, const PSKV& pskv, int priority) {
    LOG(INFO) << "Pushign compressed";
    auto push_to_servers =
      [this, key, pskv, small_buf](RunContext rctx, Engine::CallbackOnComplete cb) {
        std::cout << " sending compressed push" << std::endl;
        size_t size = small_buf.shape().Size();
        real_t* data = small_buf.data().dptr<real_t>();
        // do push. false means no delete
        ps::SArray<real_t> vals(data, size, false);
        CHECK_NOTNULL(ps_worker_)->ZPush(
          pskv.keys, vals, pskv.lens,
          static_cast<int>(DataHandleType::kCompressedPush), [cb]() { cb(); });
      };
    // acquire locks on both comm_buf and small_buf so that
    // pull (which uses comm_buf) for the same key waits till push finishes
    Engine::Get()->PushAsync(
      push_to_servers,
      pinned_ctx_,
      {small_buf.var(), comm_buf.var()},
      {},
      FnProperty::kNormal,
      priority,
      PROFILER_MESSAGE("KVStoreDistCompressedPush"));
  }

  void PushDefault(int key, const NDArray &send_buf, const PSKV& pskv, int priority) {
    LOG(INFO) << "Pushign default";
    auto push_to_servers =
        [this, key, pskv, send_buf](RunContext rctx, Engine::CallbackOnComplete cb) {
          // convert to ps keys
          size_t size = send_buf.shape().Size();
          real_t* data = send_buf.data().dptr<real_t>();
          // do push. false means no delete
          ps::SArray<real_t> vals(data, size, false);
          CHECK_NOTNULL(ps_worker_)->ZPush(
              pskv.keys, vals, pskv.lens,
              static_cast<int>(DataHandleType::kDefaultPushPull), [cb]() { cb(); });
        };
    Engine::Get()->PushAsync(
        push_to_servers,
        pinned_ctx_,
        {send_buf.var()},
        {},
        FnProperty::kNormal,
        priority,
        PROFILER_MESSAGE("KVStoreDistDefaultPush"));
  }

  // push row sparse gradient
  void PushRowSparse(int key, const NDArray &send_buf, int priority) {
    using namespace rowsparse;
    auto push_to_servers = [this, key, send_buf]
                           (RunContext rctx, Engine::CallbackOnComplete cb) {
      real_t* data = send_buf.data().dptr<real_t>();
      const int64_t num_rows = send_buf.aux_shape(kIdx)[0];
      const auto offsets = send_buf.aux_data(kIdx).dptr<int64_t>();
      const auto unit_len = send_buf.shape().ProdShape(1, send_buf.shape().ndim());
      const int64_t size = num_rows * unit_len;

       // convert to ps keys in row sparse format
      PSKV& pskv = EncodeRowSparseKey(key, size, num_rows, offsets,
                                      unit_len, send_buf.shape()[0]);
      if (this->log_verbose_row_sparse_) {
        LOG(INFO) << "worker " << get_rank() << " push lens: " << pskv.lens << " keys: "
                  << pskv.keys << " size: " << size;
      }
      ps::SArray<real_t> vals(data, size, false);
      CHECK_NOTNULL(ps_worker_)->ZPush(pskv.keys, vals, pskv.lens,
                                       static_cast<int>(DataHandleType::kRowSparsePushPull),
                                       [cb]() { cb(); });
    };
    Engine::Get()->PushAsync(
        push_to_servers,
        pinned_ctx_,
        {send_buf.var()},
        {},
        FnProperty::kNormal,
        priority,
        PROFILER_MESSAGE("KVStoreDistRowSparsePush"));
  }

  // pull row sparse weight into `recv_buf` based on indices given by `indices`
  void PullRowSparse_(const int key, const NDArray& recv_buf,
                      const NDArray& indices, int priority) {
    using namespace rowsparse;
    auto pull_from_servers = [this, key, recv_buf, indices]
      (RunContext rctx, Engine::CallbackOnComplete cb) {
      // allocate memory for the buffer
      CHECK_EQ(indices.dtype(), mshadow::kInt64);
      const TBlob idx_data = indices.data();
      size_t num_rows = idx_data.shape_.Size();
      recv_buf.CheckAndAlloc({mshadow::Shape1(num_rows)});
      real_t* data = recv_buf.data().dptr<real_t>();
      const auto offsets = idx_data.dptr<int64_t>();
      const auto unit_len = recv_buf.shape().ProdShape(1, recv_buf.shape().ndim());
      const int64_t size = num_rows * unit_len;
      // convert to ps keys in row sparse format
      PSKV& pskv = EncodeRowSparseKey(key, size, num_rows, offsets,
                                      unit_len, recv_buf.shape()[0]);
      if (this->log_verbose_row_sparse_) {
        LOG(INFO) << "worker " << get_rank() << " pull lens: " << pskv.lens << " keys: "
                  << pskv.keys << " size: " << size;
      }
      auto vals = new ps::SArray<real_t>(data, size, false);
      // copy indices to recv_buf. this needs to be done before ZPull
      // because after pull is done, the callback function returns and locks are released.
      // at this point, later functions may access the indices variable while copy happens
      mshadow::Copy(recv_buf.aux_data(kIdx).FlatTo1D<cpu, int64_t>(),
                    idx_data.FlatTo1D<cpu, int64_t>());
      CHECK_NOTNULL(ps_worker_)->ZPull(pskv.keys, vals, &pskv.lens,
                                       static_cast<int>(DataHandleType::kRowSparsePushPull),
                                       [vals, cb]() { delete vals; cb(); });
    };
    CHECK_NOTNULL(Engine::Get())->PushAsync(
      pull_from_servers,
      pinned_ctx_,
      {indices.var()},
      {recv_buf.var()},
      FnProperty::kNormal,
      priority,
      PROFILER_MESSAGE("KVStoreDistRowSparsePull"));
  }

  /**
   * \brief check if the keys are all unique
   */
  void CheckUnique(const std::vector<int>& keys) {
    auto keys_copy = keys;
    auto last = std::unique(keys_copy.begin(), keys_copy.end());
    CHECK_EQ(static_cast<size_t>(std::distance(keys_copy.begin(), last)),
             static_cast<size_t>(keys.size()));
  }

  /**
   * \brief convert to keys in ps
   */
  inline PSKV& EncodeDefaultKey(int key, size_t size, bool is_push) {
    mu_.lock();
    PSKV& pskv = ps_kv_[key];
    mu_.unlock();
    if (!pskv.keys.empty()) {
      CHECK_EQ(static_cast<size_t>(pskv.size), size) << "The value size cannot be changed";
    } else {
      auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
      int num_servers = krs.size();
      CHECK_GT(num_servers, 0);

      // a simple heuristic for load balance
      if (size < bigarray_bound_) {
        // send it to a single random picked server
        int server = (key * 9973) % num_servers;
        ps::Key ps_key = krs[server].begin() + key;
        CHECK_LT(ps_key, krs[server].end());
        pskv.keys.push_back(ps_key);
        pskv.lens.push_back(size);
        pskv.size = size;
      } else {
        // parition it to all servers
        pskv.size = 0;
        for (int i = 0; i < num_servers; ++i) {
          size_t part_size =
            static_cast<size_t>(round(static_cast<double>(size)/num_servers*(i+1))) -
            static_cast<size_t>(round(static_cast<double>(size)/num_servers*i));
          ps::Key ps_key = krs[i].begin() + key;
          CHECK_LT(ps_key, krs[i].end());
          pskv.keys.push_back(ps_key);
          pskv.lens.push_back(part_size);
          pskv.size += part_size;
        }
        CHECK_EQ(static_cast<size_t>(pskv.size), size);
      }
    }
    return pskv;
  }

  /**
   * \brief Convert to keys in ps for compressed values
   * Divides original array into equal parts for each server
   * Populates push, pull and full_pull pskv on first call
   */
  inline PSKV& EncodeCompressedKey(int key, size_t original_size, bool is_push, bool is_compressed = true) {
    mu_.lock();
    PSKV& pskv = (is_compressed && is_push) ? compr_ps_kv_[key].push :
                 ((is_compressed) ? compr_ps_kv_[key].pull : compr_ps_kv_[key].full_pull);
    mu_.unlock();

    size_t pull_compr_size = gradient_compression_->GetServerRecompressedSize(original_size);
    size_t pull_size = (is_compressed) ? pull_compr_size : original_size;
    if (!pskv.keys.empty()) {
      if (!is_push) {
        CHECK_EQ(static_cast<size_t >(pskv.size), pull_size)
          << ps::MyRank() <<": The value size can't be changed. is_push is "<<is_push
          <<"; and is_compressed is "<<is_compressed << " for key "<<key;
      }
    } else {
      auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
      int num_servers = krs.size();
      int num_workers = ps::NumWorkers();
      CHECK_GT(num_servers, 0);

      // populate both pull and push pskvs
      // push pskv has sizes corresponding to compressed data
      // pull pskv has decompressed sizes for parts in compr_push_pskv
      mu_.lock();
      PSKV& compr_pull_pskv = compr_ps_kv_[key].pull;
      PSKV& compr_push_pskv = compr_ps_kv_[key].push;
      PSKV& full_pull_pskv = compr_ps_kv_[key].full_pull;
      mu_.unlock();

      if (original_size < bigarray_bound_) {
        // a simple heuristic for load balancing
        // send it to a single random picked server
        int server = (key * 9973) % num_servers;
        ps::Key ps_key = krs[server].begin() + key;
        CHECK_LT(ps_key, krs[server].end());
        size_t push_compr_size = gradient_compression_->GetCompressedSize(original_size);

        // meta info to let server know about original_size
        compr_push_pskv.keys.push_back(krs[server].begin() + original_size);
        compr_push_pskv.lens.push_back(0);
        // data
        compr_push_pskv.keys.push_back(ps_key);
        compr_push_pskv.lens.push_back(push_compr_size);
        compr_push_pskv.size = push_compr_size;

        compr_pull_pskv.keys.push_back(ps_key);
        compr_pull_pskv.lens.push_back(pull_compr_size);
        compr_pull_pskv.size = pull_compr_size;

        full_pull_pskv.keys.push_back(ps_key);
        full_pull_pskv.lens.push_back(original_size);
        full_pull_pskv.size = original_size;
      } else {
        // partition it to all servers
        compr_push_pskv.size = 0;
        full_pull_pskv.size = 0;
        compr_pull_pskv.size = 0;

        int block_size = gradient_compression_->GetServerCompressionBlockSize(num_workers);
        int num_blocks = pull_compr_size / block_size;
        size_t num_bits = gradient_compression_->GetServerCompressionNumBits(num_workers);
        for (int i = 0; i < num_servers; ++i) {
          size_t push_part, part_orig, pull_part, part_blocks;
          if (i == num_servers-1) {
            pull_part = pull_compr_size - compr_pull_pskv.size;
            part_orig = original_size - full_pull_pskv.size;
            push_part = gradient_compression_->GetCompressedSize(part_orig);
          } else {
            part_blocks = static_cast<size_t> (round(static_cast<double>(num_blocks)/num_servers*(i+1))) -
                          static_cast<size_t> (round(static_cast<double>(num_blocks)/num_servers*(i)));
            pull_part = part_blocks * block_size;

            part_orig = pull_part * 32 / num_bits;
            CHECK_EQ((pull_part * 32) % num_bits, 0);
            push_part = gradient_compression_->GetCompressedSize(part_orig);
          }

          // meta info
          ps::Key ps_key_dummy = krs[i].begin() + part_orig;
          CHECK_LT(ps_key_dummy, krs[i].end());
          compr_push_pskv.keys.push_back(ps_key_dummy);
          compr_push_pskv.lens.push_back(0);

          // data
          ps::Key ps_key = krs[i].begin() + key;
          CHECK_LT(ps_key, krs[i].end());
          compr_push_pskv.keys.push_back(ps_key);
          compr_pull_pskv.keys.push_back(ps_key);
          full_pull_pskv.keys.push_back(ps_key);

          // compr_push_pskv stores lengths of compressed blocks
          compr_push_pskv.lens.push_back(push_part);
          compr_push_pskv.size += push_part;

          compr_pull_pskv.lens.push_back(pull_part);
          compr_pull_pskv.size += pull_part;

          full_pull_pskv.lens.push_back(part_orig);
          full_pull_pskv.size += part_orig;
        }
        CHECK_EQ(static_cast<size_t>(compr_pull_pskv.size), pull_compr_size);
        CHECK_EQ(static_cast<size_t>(full_pull_pskv.size), original_size);
        CHECK_EQ(compr_push_pskv.lens.size(), num_servers*2);
        // to check that sizes haven't changed for next operations on same key
        if (gradient_compression_->get_server_compression_type() == CompressionType::kNone) {
          CHECK_EQ(static_cast<size_t>(compr_pull_pskv.size), original_size);
        }
      }
    }
    return pskv;
  }

  // Note: this encoding method for row sparse keys doesn't allow cross-layer batching
  inline PSKV& EncodeRowSparseKey(const int key, const int64_t size, const int64_t num_rows,
                                  const int64_t *offsets, const size_t unit_len,
                                  const int64_t total_num_rows) {
    using namespace common;
    mu_.lock();
    PSKV& pskv = ps_kv_[key];
    mu_.unlock();
    pskv.keys.clear();
    pskv.lens.clear();
    // TODO(haibin) cache this information
    auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
    int num_servers = krs.size();
    CHECK_GT(num_servers, 0);

    if (total_num_rows * unit_len >= bigarray_bound_) {
      pskv.size = 0;
      int64_t start_row = 0;
      // parition it to all servers
      for (int i = 0; i < num_servers; ++i) {
        ps::Key master_key = krs[i].begin() + key;
        pskv.keys.push_back(master_key);
        pskv.lens.push_back(0);
        if (offsets && size > 0) {
          // calculate partition ranges
          int64_t part_num_rows =
            llround(static_cast<double>(total_num_rows) / num_servers * (i + 1)) -
            llround(static_cast<double>(total_num_rows) / num_servers * i);
          auto end_row = start_row + part_num_rows;
          // search for offsets in [start_row, end_row)
          auto lb = std::lower_bound(offsets, offsets + num_rows, start_row);
          auto ub = std::upper_bound(offsets, offsets + num_rows, end_row - 1);
          for (auto offset = lb; offset < ub; offset++) {
            ps::Key ps_key = krs[i].begin() + key + (*offset - start_row);
            CHECK_LT(ps_key, krs[i].end());
            pskv.keys.push_back(ps_key);
            pskv.lens.push_back(unit_len);
            pskv.size += unit_len;
          }
          start_row = end_row;
        }
      }
      CHECK_EQ(static_cast<size_t>(pskv.size), size);
    } else {
      // send it to a single random picked server
      int server = (key * 9973) % num_servers;
      ps::Key master_key = krs[server].begin() + key;
      pskv.keys.push_back(master_key);
      pskv.lens.push_back(0);
      for (int64_t i = 0; i < num_rows; i++) {
        ps::Key ps_key = krs[server].begin() + key + offsets[i];
        CHECK_LT(ps_key, krs[server].end());
        pskv.keys.push_back(ps_key);
        pskv.lens.push_back(unit_len);
      }
      pskv.size = size;
    }
    return pskv;
  }

  /**
   * \brief for worker to push and pull data
   */
  ps::KVWorker<real_t>* ps_worker_;
  /**
   * \brief the server handle
   */
  KVStoreDistServer* server_;
  bool sync_mode_;
  /**
   * \brief threshold for partition
   */
  size_t bigarray_bound_;
  /**
   * \brief store_ contains the value at kvstore for each key
   */
  std::unordered_map<int, NDArray> store_;
  /**
   * \brief buffer for non-compressed data.
   */
  std::unordered_map<int, NDArray> send_buf_;
  std::unordered_map<int, NDArray> recv_buf_;
  /**
   * \brief buffer for compressed data
   * Used when gradient compression is active
   */
  std::unordered_map<int, NDArray> send_compr_buf_;
  std::unordered_map<int, NDArray> recv_compr_buf_;
  std::unordered_map<int, NDArray> decomp_buf_;

  std::unordered_map<int, bool> first_pull_done_;
  /**
   * \brief residual buffer to accumulate quantization error
   * during gradient compression
   */
  std::unordered_map<int, NDArray> residual_;
  bool log_verbose_row_sparse_;
  bool log_verbose_gc_;
};

}  // namespace kvstore
}  // namespace mxnet


#endif  // MXNET_KVSTORE_KVSTORE_DIST_H_
