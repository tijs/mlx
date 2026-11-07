// Copyright © 2023-2024 Apple Inc.
#include "mlx/backend/metal/allocator.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/resident.h"
#include "mlx/memory.h"

#include <execinfo.h>
#include <mach/vm_page_size.h>
#include <unistd.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace mlx::core {

constexpr size_t resource_options =
    MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked;

namespace allocator {

Allocator& allocator() {
  return metal::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  auto* buf = static_cast<MTL::Buffer*>(ptr_);
  assert(buf->storageMode() != MTL::StorageModePrivate);
  return buf->contents();
}

bool can_reuse_alien_buffer(void* ptr) {
  if (!ptr) {
    return true;
  }
  auto* buf = static_cast<MTL::Buffer*>(ptr);
  return buf->storageMode() != MTL::StorageModePrivate;
}

} // namespace allocator

namespace metal {

MetalAllocator::MetalAllocator(Device& d)
    : device_(d.mtl_device()),
      residency_sets_(d.residency_sets()),
      buffer_cache_(
          vm_page_size,
          [](MTL::Buffer* buf) { return buf->length(); },
          [this](MTL::Buffer* buf) {
            if (!buf->heap()) {
              residency_sets_.erase(buf);
            }
            auto pool = metal::new_scoped_memory_pool();
            buf->release();
          }) {
  const auto& info = gpu::device_info(0);
  auto memsize = std::get<size_t>(info.at("memory_size"));
  auto max_rec_size =
      std::get<size_t>(info.at("max_recommended_working_set_size"));
  resource_limit_ = std::get<size_t>(info.at("resource_limit"));
  block_limit_.store(
      std::min(1.5 * max_rec_size, 0.95 * memsize), std::memory_order_relaxed);
  gc_limit_ = std::min(
      static_cast<size_t>(0.95 * max_rec_size),
      block_limit_.load(std::memory_order_relaxed));
  max_pool_size_ = block_limit_.load(std::memory_order_relaxed);
  bool is_vm = std::get<std::string>(info.at("device_name")) ==
      "Apple Paravirtual device";
  if (is_vm) {
    return;
  }
  auto pool = metal::new_scoped_memory_pool();
  auto heap_desc = MTL::HeapDescriptor::alloc()->init()->autorelease();
  heap_desc->setResourceOptions(resource_options);
  heap_desc->setSize(heap_size_);
  heap_ = NS::TransferPtr(device_->newHeap(heap_desc));
  residency_sets_.insert(heap_.get());
}

MetalAllocator::~MetalAllocator() = default;

size_t MetalAllocator::set_cache_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(limit, max_pool_size_);
  return limit;
};

size_t MetalAllocator::set_memory_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  limit = block_limit_.exchange(limit, std::memory_order_relaxed);
  gc_limit_ = std::min(
      block_limit_.load(std::memory_order_relaxed),
      static_cast<size_t>(0.95 * device_->recommendedMaxWorkingSetSize()));
  return limit;
};

size_t MetalAllocator::get_memory_limit() {
  return block_limit_.load(std::memory_order_relaxed);
}

size_t MetalAllocator::set_wired_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(limit, wired_limit_);
  residency_sets_.resize(wired_limit_);
  return limit;
};

// 2026-04-30 (osaurus stability work): env-gated tracer for large
// allocations. Bug 2 (over-cap hybrid prompt → 154 GB metal::malloc on
// some hosts) needs a way to pin the *exact* call site of an
// over-sized allocation request before designing a clamp. Set
// OSAURUS_MLX_MALLOC_TRACE=1 to log every malloc whose requested size
// is >= OSAURUS_MLX_MALLOC_TRACE_BYTES (default 1 GiB) along with a
// symbolicated C++ backtrace to stderr. Off by default — the wrapper
// adds nothing to the hot path when the env is unset (single
// once-per-process getenv read, then a cheap size compare). The
// backtrace is written under a static mutex so concurrent decode
// threads don't interleave frames mid-line.
namespace {
struct MallocTraceConfig {
  bool enabled = false;
  size_t threshold_bytes = 1024ull * 1024ull * 1024ull; // 1 GiB
};

const MallocTraceConfig& malloc_trace_config() {
  static MallocTraceConfig cfg = []() {
    MallocTraceConfig c;
    if (const char* v = std::getenv("OSAURUS_MLX_MALLOC_TRACE")) {
      c.enabled = (v[0] == '1');
    }
    if (const char* v = std::getenv("OSAURUS_MLX_MALLOC_TRACE_BYTES")) {
      char* end = nullptr;
      unsigned long long n = std::strtoull(v, &end, 10);
      if (end != v && n > 0) {
        c.threshold_bytes = static_cast<size_t>(n);
      }
    }
    if (c.enabled) {
      std::fprintf(
          stderr,
          "[osaurus.malloc-trace] enabled, threshold=%zu bytes\n",
          c.threshold_bytes);
    }
    return c;
  }();
  return cfg;
}

void log_large_alloc(size_t size) {
  static std::mutex log_mu;
  std::lock_guard<std::mutex> lk(log_mu);
  std::fprintf(
      stderr,
      "[osaurus.malloc-trace] metal::malloc requested %.2f GiB (%zu bytes)\n",
      static_cast<double>(size) / (1024.0 * 1024.0 * 1024.0),
      size);
  void* frames[32];
  int n = ::backtrace(frames, 32);
  if (n > 0) {
    char** syms = ::backtrace_symbols(frames, n);
    if (syms) {
      // Skip frame 0 (this function) and frame 1 (caller inside
      // MetalAllocator::malloc).
      for (int i = 2; i < n; ++i) {
        std::fprintf(
            stderr, "[osaurus.malloc-trace]   #%-2d %s\n", i - 2, syms[i]);
      }
      std::free(syms);
    }
  }
  std::fflush(stderr);
}
} // namespace

Buffer MetalAllocator::malloc(size_t size) {
  // Metal doesn't like empty buffers
  if (size == 0) {
    return Buffer{nullptr};
  }

  // 2026-04-30 (osaurus stability work): trace large allocations when
  // OSAURUS_MLX_MALLOC_TRACE=1. See the header comment on
  // malloc_trace_config above for details. Zero overhead when env unset.
  {
    const auto& cfg = malloc_trace_config();
    if (cfg.enabled && size >= cfg.threshold_bytes) {
      log_large_alloc(size);
    }
  }

  // More helpful message if maximum buffer length is exceeded
  if (size > device_->maxBufferLength()) {
    std::ostringstream msg;
    msg << "[metal::malloc] Attempting to allocate " << size
        << " bytes which is greater than"
        << " the maximum allowed buffer size of " << device_->maxBufferLength()
        << " bytes.";
    throw std::runtime_error(msg.str());
  }

  // Align up memory
  if (size > vm_page_size) {
    size = vm_page_size * ((size + vm_page_size - 1) / vm_page_size);
  }

  // Try the cache
  std::unique_lock lk(mutex_);
  MTL::Buffer* buf = buffer_cache_.reuse_from_cache(size);
  if (!buf) {
    size_t mem_required =
        get_active_memory() + buffer_cache_.cache_size() + size;

    // If we have a lot of memory pressure try to reclaim memory from the cache
    if (mem_required >= gc_limit_ || num_resources_ >= resource_limit_) {
      num_resources_ -=
          buffer_cache_.release_cached_buffers(mem_required - gc_limit_);
    }

    // Allocate new buffer if needed
    if (num_resources_ >= resource_limit_) {
      std::ostringstream msg;
      msg << "[metal::malloc] Resource limit (" << resource_limit_
          << ") exceeded.";
      throw std::runtime_error(msg.str());
    }
    lk.unlock();
    if (size < small_size_ && heap_) {
      buf = heap_->newBuffer(size, resource_options);
    }
    if (!buf) {
      buf = device_->newBuffer(size, resource_options);
    }
    if (!buf) {
      std::ostringstream msg;
      msg << "[malloc] Unable to allocate " << size << " bytes.";
      throw std::runtime_error(msg.str());
    }
    lk.lock();
    num_resources_++;
    if (!buf->heap()) {
      residency_sets_.insert(buf);
    }
  }

  active_memory_.store(
      active_memory_.load(std::memory_order_relaxed) + buf->length(),
      std::memory_order_relaxed);
  peak_memory_.store(
      std::max(
          peak_memory_.load(std::memory_order_relaxed),
          active_memory_.load(std::memory_order_relaxed)),
      std::memory_order_relaxed);

  // Maintain the cache below the requested limit
  if (buffer_cache_.cache_size() > max_pool_size_) {
    num_resources_ -= buffer_cache_.release_cached_buffers(
        buffer_cache_.cache_size() - max_pool_size_);
  }

  Buffer buffer(static_cast<void*>(buf));
#ifndef NDEBUG
  track_allocation(buffer.raw_ptr(), size);
#endif
  return buffer;
}

void MetalAllocator::clear_cache() {
  std::unique_lock lk(mutex_);
  num_resources_ -= buffer_cache_.clear();
}

void MetalAllocator::free(Buffer buffer) {
  auto buf = static_cast<MTL::Buffer*>(buffer.ptr());
  if (buf == nullptr) {
    return;
  }

#ifndef NDEBUG
  untrack_allocation(buffer.raw_ptr());
#endif

  std::unique_lock lk(mutex_);
  active_memory_.store(
      active_memory_.load(std::memory_order_relaxed) - buf->length(),
      std::memory_order_relaxed);
  if (buf->length() <= max_pool_size_) {
    // Keep recent shapes instead of filling the pool with stale buffers.
    const size_t available = max_pool_size_ - buf->length();
    if (buffer_cache_.cache_size() > available) {
      num_resources_ -= buffer_cache_.release_cached_buffers(
          buffer_cache_.cache_size() - available);
    }
    buffer_cache_.recycle_to_cache(buf);
  } else {
    num_resources_--;
    if (!buf->heap()) {
      residency_sets_.erase(buf);
    }
    lk.unlock();
    auto pool = metal::new_scoped_memory_pool();
    buf->release();
  }
}

size_t MetalAllocator::size(Buffer buffer) const {
  return static_cast<MTL::Buffer*>(buffer.ptr())->length();
}

Buffer MetalAllocator::make_buffer(void* ptr, size_t size) {
  auto buf = device_->newBuffer(ptr, size, resource_options, nullptr);
  if (!buf) {
    return Buffer{nullptr};
  }
  std::unique_lock lk(mutex_);
  residency_sets_.insert(buf);
  active_memory_.store(
      active_memory_.load(std::memory_order_relaxed) + buf->length(),
      std::memory_order_relaxed);
  peak_memory_.store(
      std::max(
          peak_memory_.load(std::memory_order_relaxed),
          active_memory_.load(std::memory_order_relaxed)),
      std::memory_order_relaxed);
  num_resources_++;
  return Buffer{static_cast<void*>(buf)};
}

void MetalAllocator::release(Buffer buffer) {
  auto buf = static_cast<MTL::Buffer*>(buffer.ptr());
  if (buf == nullptr) {
    return;
  }
  std::unique_lock lk(mutex_);
  active_memory_.store(
      active_memory_.load(std::memory_order_relaxed) - buf->length(),
      std::memory_order_relaxed);
  num_resources_--;
  residency_sets_.erase(buf);
  lk.unlock();
  auto pool = metal::new_scoped_memory_pool();
  buf->release();
}

MetalAllocator& allocator() {
  // By creating the |allocator_| on heap, the destructor of MetalAllocator
  // will not be called on exit and buffers in the cache will be leaked. This
  // can save some time at program exit.
  static MetalAllocator* allocator_ =
      new MetalAllocator(device(mlx::core::Device::gpu));
  return *allocator_;
}

} // namespace metal

size_t set_cache_limit(size_t limit) {
  return metal::allocator().set_cache_limit(limit);
}
size_t set_memory_limit(size_t limit) {
  return metal::allocator().set_memory_limit(limit);
}
size_t get_memory_limit() {
  return metal::allocator().get_memory_limit();
}
size_t set_wired_limit(size_t limit) {
  if (limit > std::get<size_t>(
                  gpu::device_info(0).at("max_recommended_working_set_size"))) {
    throw std::invalid_argument(
        "[metal::set_wired_limit] Setting a wired limit larger than "
        "the maximum working set size is not allowed.");
  }
  return metal::allocator().set_wired_limit(limit);
}
size_t get_active_memory() {
  return metal::allocator().get_active_memory();
}
size_t get_peak_memory() {
  return metal::allocator().get_peak_memory();
}
void reset_peak_memory() {
  metal::allocator().reset_peak_memory();
}
size_t get_cache_memory() {
  return metal::allocator().get_cache_memory();
}
void clear_cache() {
  return metal::allocator().clear_cache();
}

} // namespace mlx::core
