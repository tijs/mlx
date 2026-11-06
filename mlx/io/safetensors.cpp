// Copyright © 2023 Apple Inc.

#include <json.hpp>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <sstream>
#include <stack>
#include <string>
#include <tuple>
#include <unordered_set>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "mlx/allocator.h"
#include "mlx/backend/cuda/cuda.h"
#include "mlx/io.h"
#include "mlx/io/load.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

using json = nlohmann::json;

#define ST_F16 "F16"
#define ST_BF16 "BF16"
#define ST_F32 "F32"

#define ST_BOOL "BOOL"
#define ST_I8 "I8"
#define ST_I16 "I16"
#define ST_I32 "I32"
#define ST_I64 "I64"
#define ST_U8 "U8"
#define ST_U16 "U16"
#define ST_U32 "U32"
#define ST_U64 "U64"
#define ST_F8_E4M3 "F8_E4M3"
#define ST_F8_E8M0 "F8_E8M0"

// Note: Complex numbers aren't in the spec yet so this could change -
// https://github.com/huggingface/safetensors/issues/389
#define ST_C64 "C64"

namespace mlx::core {

std::string dtype_to_safetensor_str(Dtype t) {
  switch (t) {
    case float32:
      return ST_F32;
    case bfloat16:
      return ST_BF16;
    case float16:
      return ST_F16;
    case int64:
      return ST_I64;
    case int32:
      return ST_I32;
    case int16:
      return ST_I16;
    case int8:
      return ST_I8;
    case uint64:
      return ST_U64;
    case uint32:
      return ST_U32;
    case uint16:
      return ST_U16;
    case uint8:
      return ST_U8;
    case bool_:
      return ST_BOOL;
    case complex64:
      return ST_C64;
    default:
      throw std::runtime_error("[save_safetensors] received invalid dtype.");
  }
}

Dtype dtype_from_safetensor_str(std::string_view str) {
  if (str == ST_F32) {
    return float32;
  } else if (str == ST_F16) {
    return float16;
  } else if (str == ST_BF16) {
    return bfloat16;
  } else if (str == ST_I64) {
    return int64;
  } else if (str == ST_I32) {
    return int32;
  } else if (str == ST_I16) {
    return int16;
  } else if (str == ST_I8) {
    return int8;
  } else if (str == ST_U64) {
    return uint64;
  } else if (str == ST_U32) {
    return uint32;
  } else if (str == ST_U16) {
    return uint16;
  } else if (str == ST_U8) {
    return uint8;
  } else if (str == ST_BOOL) {
    return bool_;
  } else if (str == ST_C64) {
    return complex64;
  } else if (str == ST_F8_E4M3) {
    return uint8;
  } else if (str == ST_F8_E8M0) {
    return uint8;
  } else {
    std::ostringstream msg;
    msg << "[safetensor] unsupported dtype " << str;
    throw std::runtime_error(msg.str());
  }
}

#ifndef _WIN32
namespace {

bool env_truthy(const char* key) {
  auto raw = std::getenv(key);
  if (!raw) {
    return false;
  }
  std::string value(raw);
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
  return value == "1" || value == "true" || value == "on" || value == "yes";
}

bool mmap_debug_enabled() {
  return env_truthy("MLX_SAFETENSORS_MMAP_DEBUG") ||
      env_truthy("VMLINUX_MMAP_SAFETENSORS_DEBUG");
}

std::optional<std::string> env_lower(const char* key) {
  auto raw = std::getenv(key);
  if (!raw) {
    return std::nullopt;
  }
  std::string value(raw);
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
  return value;
}

bool mmap_tensor_buffers_enabled() {
  return env_truthy("MLX_SAFETENSORS_MMAP_TENSOR_BUFFERS") ||
      env_truthy("VMLINUX_MMAP_SAFETENSORS_TENSOR_BUFFERS") ||
      env_truthy("VMLX_MMAP_SAFETENSORS_TENSOR_BUFFERS");
}

int32_t env_int_clamped(
    const char* key,
    int32_t default_value,
    int32_t min_value,
    int32_t max_value) {
  auto raw = std::getenv(key);
  if (!raw) {
    return default_value;
  }
  char* end = nullptr;
  const auto parsed = std::strtol(raw, &end, 10);
  if (end == raw) {
    return default_value;
  }
  return std::max<int32_t>(
      min_value, std::min<int32_t>(max_value, static_cast<int32_t>(parsed)));
}

bool mmap_start_cold_enabled() {
  return env_truthy("MLX_SAFETENSORS_MMAP_START_COLD") ||
      env_truthy("VMLINUX_MMAP_SAFETENSORS_START_COLD");
}

int32_t mmap_start_cold_pct() {
  if (std::getenv("MLX_SAFETENSORS_MMAP_COLD_PCT")) {
    return env_int_clamped("MLX_SAFETENSORS_MMAP_COLD_PCT", 70, 0, 100);
  }
  return env_int_clamped("VMLINUX_MMAP_SAFETENSORS_COLD_PCT", 70, 0, 100);
}

enum class MmapColdAdvice {
  dont_need,
  page_out,
  invalidate,
};

MmapColdAdvice mmap_cold_advice() {
  auto value = env_lower("MLX_SAFETENSORS_MMAP_COLD_ADVICE");
  if (!value) {
    value = env_lower("VMLINUX_MMAP_SAFETENSORS_COLD_ADVICE");
  }
  if (!value) {
    return MmapColdAdvice::dont_need;
  }
  std::replace(value->begin(), value->end(), '-', '_');
  if (*value == "pageout" || *value == "page_out") {
    return MmapColdAdvice::page_out;
  }
  if (*value == "force" || *value == "invalidate" || *value == "msync" ||
      *value == "msync_invalidate") {
    return MmapColdAdvice::invalidate;
  }
  return MmapColdAdvice::dont_need;
}

int mmap_cold_madvise_value(MmapColdAdvice advice) {
  switch (advice) {
    case MmapColdAdvice::page_out:
#ifdef MADV_PAGEOUT
      return MADV_PAGEOUT;
#else
      return MADV_DONTNEED;
#endif
    case MmapColdAdvice::invalidate:
    case MmapColdAdvice::dont_need:
      return MADV_DONTNEED;
  }
  throw std::invalid_argument("[safetensors] Invalid cold-memory advice.");
}

struct MmapShard {
  void* base{nullptr};
  size_t size{0};
  size_t tracked_buffer_bytes{0};
  std::string path;

  MmapShard(void* base, size_t size, std::string path)
      : base(base), size(size), path(std::move(path)) {}

  MmapShard(const MmapShard&) = delete;
  MmapShard& operator=(const MmapShard&) = delete;

  ~MmapShard() {
    if (base && base != MAP_FAILED && size > 0) {
      munmap(base, size);
    }
  }
};

struct ParsedRoutedName {
  int32_t layer;
  int32_t expert;
  bool stacked;
};

std::optional<ParsedRoutedName> match_routed_name(const std::string& name) {
  static const std::string vl_prefix = R"((?:(?:model|language_model)\.)*)";
  static const std::vector<std::regex> per_expert = {
      std::regex(
          "^" + vl_prefix +
          R"(layers\.(\d+)\.mlp\.experts\.(\d+)\.(?:gate|up|down)_proj\.(?:weight|tq_packed|tq_norms)$)"),
      std::regex(
          R"(^layers\.(\d+)\.ffn\.experts\.(\d+)\.(?:w[123]|(?:gate|up|down)_proj)\.(?:tq_packed|tq_norms)$)"),
      std::regex(
          R"(^layers\.(\d+)\.ffn\.experts\.(\d+)\.(?:w[123]|(?:gate|up|down)_proj)\.weight$)"),
      std::regex(
          "^" + vl_prefix +
          R"(layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w[123]\.(?:tq_packed|tq_norms)$)"),
      std::regex(
          "^" + vl_prefix +
          R"(layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w[123]\.weight$)"),
      std::regex(
          R"(^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(?:gate|up|down)_proj\.(?:tq_packed|tq_norms)$)"),
      std::regex(
          R"(^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(?:gate|up|down)_proj\.weight$)")};
  static const std::vector<std::regex> stacked = {
      std::regex(
          "^" + vl_prefix +
          R"(layers\.(\d+)\.mlp\.switch_mlp\.(?:gate|up|down)_proj\.weight$)"),
      std::regex(
          "^" + vl_prefix +
          R"(layers\.(\d+)\.mlp\.experts\.(?:gate_up_proj|down_proj|gate_proj|up_proj)\.tq_packed$)"),
      std::regex(
          "^" + vl_prefix +
          R"(layers\.(\d+)\.mlp\.experts\.(?:gate_up_proj|down_proj|gate_proj|up_proj)\.weight$)"),
      std::regex(
          "^" + vl_prefix +
          R"(layers\.(\d+)\.mlp\.switch_mlp\.(?:gate|up|down)_proj\.(?:tq_packed|tq_norms)$)"),
      std::regex(
          "^" + vl_prefix +
          R"(layers\.(\d+)\.(?:mlp\.)?zaya_block\.experts\.switch_mlp\.(?:gate|up|down)_proj\.(?:tq_packed|tq_norms)$)"),
      std::regex(
          "^" + vl_prefix +
          R"(layers\.(\d+)\.switch_mlp\.(?:gate|up|down)_proj\.(?:weight|scales|biases|tq_packed|tq_norms)$)"),
      std::regex(
          "^" + vl_prefix +
          R"(layers\.(\d+)\.block_sparse_moe\.switch_mlp\.(?:gate|up|down)_proj\.(?:weight|scales|biases|tq_packed|tq_norms)$)"),
      std::regex(
          R"(^backbone\.layers\.(\d+)\.mixer\.switch_mlp\.fc[12]\.(?:weight|tq_packed|tq_norms)$)"),
      std::regex(
          R"(^backbone\.layers\.(\d+)\.mixer\.switch_mlp\.(?:gate|up|down)_proj\.weight$)"),
      std::regex(
          R"(^layers\.(\d+)\.ffn\.switch_mlp\.(?:gate|up|down)_proj\.(?:weight|scales|biases|tq_packed|tq_norms)$)")};

  std::smatch match;
  for (const auto& regex : per_expert) {
    if (std::regex_match(name, match, regex) && match.size() >= 3) {
      return ParsedRoutedName{
          static_cast<int32_t>(std::stoi(match[1].str())),
          static_cast<int32_t>(std::stoi(match[2].str())),
          false};
    }
  }
  for (const auto& regex : stacked) {
    if (std::regex_match(name, match, regex) && match.size() >= 2) {
      return ParsedRoutedName{
          static_cast<int32_t>(std::stoi(match[1].str())), 0, true};
    }
  }
  return std::nullopt;
}

std::optional<int32_t> match_layer_name(const std::string& name) {
  static const std::regex layer_regex(
      R"(^(?:(?:model|language_model)\.)*layers\.(\d+)\.)");
  std::smatch match;
  if (std::regex_search(name, match, layer_regex) && match.size() >= 2) {
    return static_cast<int32_t>(std::stoi(match[1].str()));
  }
  return std::nullopt;
}

struct MmapTensorRegion {
  std::weak_ptr<MmapShard> shard;
  int32_t layer;
  int32_t expert;
  size_t offset;
  size_t length;
};

struct LiveMmapTensorRegion {
  std::shared_ptr<MmapShard> shard;
  int32_t layer;
  int32_t expert;
  size_t offset;
  size_t length;
};

class SafetensorsMmapRegistry {
 public:
  static SafetensorsMmapRegistry& instance() {
    static SafetensorsMmapRegistry registry;
    return registry;
  }

  void register_region(
      const std::shared_ptr<MmapShard>& shard,
      int32_t layer,
      int32_t expert,
      size_t offset,
      size_t length) {
    std::lock_guard<std::mutex> lock(mutex_);
    regions_.push_back(
        MmapTensorRegion{
            std::weak_ptr<MmapShard>(shard), layer, expert, offset, length});
  }

  void register_layer_region(
      const std::shared_ptr<MmapShard>& shard,
      int32_t layer,
      size_t offset,
      size_t length) {
    std::lock_guard<std::mutex> lock(mutex_);
    layer_regions_.push_back(
        MmapTensorRegion{
            std::weak_ptr<MmapShard>(shard), layer, -1, offset, length});
  }

  int64_t advise_routed(int32_t advice, int32_t cold_pct) {
    auto regions = live_regions();
    if (regions.empty()) {
      return 0;
    }

    if (advice != 0) {
      return advise_regions(regions, advice);
    }

    cold_pct = std::max<int32_t>(0, std::min<int32_t>(100, cold_pct));
    if (cold_pct == 0) {
      return 0;
    }
    if (cold_pct == 100) {
      return advise_regions(regions, advice);
    }

    std::sort(
        regions.begin(), regions.end(), [](const auto& lhs, const auto& rhs) {
          return std::tie(lhs.layer, lhs.expert, lhs.offset, lhs.length) <
              std::tie(rhs.layer, rhs.expert, rhs.offset, rhs.length);
        });

    std::unordered_map<int32_t, std::vector<int32_t>> experts_by_layer;
    for (const auto& region : regions) {
      auto& experts = experts_by_layer[region.layer];
      if (std::find(experts.begin(), experts.end(), region.expert) ==
          experts.end()) {
        experts.push_back(region.expert);
      }
    }
    std::unordered_set<int64_t> cold_pairs;
    for (auto& [layer, experts] : experts_by_layer) {
      std::sort(experts.begin(), experts.end());
      const auto hot_pct = 100 - cold_pct;
      const auto hot_count = static_cast<size_t>(
          (static_cast<int64_t>(experts.size()) * hot_pct + 99) / 100);
      for (size_t i = hot_count; i < experts.size(); ++i) {
        cold_pairs.insert(pair_key(layer, experts[i]));
      }
    }

    std::vector<LiveMmapTensorRegion> selected;
    selected.reserve(regions.size());
    for (const auto& region : regions) {
      if (cold_pairs.contains(pair_key(region.layer, region.expert))) {
        selected.push_back(region);
      }
    }
    return advise_regions(selected, advice);
  }

  int64_t advise_experts(
      int32_t advice,
      const int32_t* layers,
      const int32_t* experts,
      int64_t count) {
    if (count <= 0 || layers == nullptr || experts == nullptr) {
      return 0;
    }
    std::unordered_set<int64_t> requested;
    requested.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
      requested.insert(pair_key(layers[i], experts[i]));
    }

    std::vector<LiveMmapTensorRegion> selected;
    for (auto& region : live_regions()) {
      if (requested.contains(pair_key(region.layer, region.expert))) {
        selected.push_back(std::move(region));
      }
    }
    return advise_regions(selected, advice);
  }

  int64_t advise_layer(int32_t advice, int32_t layer) {
    std::vector<LiveMmapTensorRegion> selected;
    for (auto& region : live_layer_regions()) {
      if (region.layer == layer) {
        selected.push_back(std::move(region));
      }
    }
    return advise_regions(selected, advice);
  }

  int64_t tracked_buffer_bytes() {
    std::lock_guard<std::mutex> lock(mutex_);
    int64_t total = 0;
    std::unordered_set<const MmapShard*> seen;
    auto visit = [&](std::vector<MmapTensorRegion>& regions) {
      auto write = regions.begin();
      for (auto read = regions.begin(); read != regions.end(); ++read) {
        if (auto shard = read->shard.lock()) {
          if (seen.insert(shard.get()).second) {
            total += static_cast<int64_t>(shard->tracked_buffer_bytes);
          }
          *write++ = *read;
        }
      }
      regions.erase(write, regions.end());
    };
    visit(regions_);
    visit(layer_regions_);
    return total;
  }

 private:
  static int64_t pair_key(int32_t layer, int32_t expert) {
    return (static_cast<int64_t>(layer) << 32) ^ static_cast<uint32_t>(expert);
  }

  std::vector<LiveMmapTensorRegion> live_regions() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<LiveMmapTensorRegion> live;
    live.reserve(regions_.size());
    auto write = regions_.begin();
    for (auto read = regions_.begin(); read != regions_.end(); ++read) {
      if (auto shard = read->shard.lock()) {
        live.push_back(
            LiveMmapTensorRegion{
                std::move(shard),
                read->layer,
                read->expert,
                read->offset,
                read->length});
        *write++ = *read;
      }
    }
    regions_.erase(write, regions_.end());
    return live;
  }

  std::vector<LiveMmapTensorRegion> live_layer_regions() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<LiveMmapTensorRegion> live;
    live.reserve(layer_regions_.size());
    auto write = layer_regions_.begin();
    for (auto read = layer_regions_.begin(); read != layer_regions_.end();
         ++read) {
      if (auto shard = read->shard.lock()) {
        live.push_back(
            LiveMmapTensorRegion{
                std::move(shard),
                read->layer,
                read->expert,
                read->offset,
                read->length});
        *write++ = *read;
      }
    }
    layer_regions_.erase(write, layer_regions_.end());
    return live;
  }

  static int64_t advise_regions(
      const std::vector<LiveMmapTensorRegion>& regions,
      int32_t advice) {
    int64_t advised = 0;
    for (const auto& region : regions) {
      advised += advise_region(region, advice);
    }
    return advised;
  }

  static int64_t advise_region(
      const LiveMmapTensorRegion& region,
      int32_t advice) {
    if (!region.shard || region.length == 0 ||
        region.offset >= region.shard->size) {
      return 0;
    }
    const auto clamped_length =
        std::min(region.length, region.shard->size - region.offset);
    const auto page = static_cast<uintptr_t>(getpagesize());
    const auto base = reinterpret_cast<uintptr_t>(region.shard->base);
    const auto start = base + region.offset;
    const auto end = start + clamped_length;
    const auto aligned_start = start & ~(page - 1);
    const auto aligned_end = (end + page - 1) & ~(page - 1);
    if (aligned_end <= aligned_start) {
      return 0;
    }
    void* const aligned_address = reinterpret_cast<void*>(aligned_start);
    const auto aligned_length = aligned_end - aligned_start;
    if (advice == 1) {
      if (madvise(aligned_address, aligned_length, MADV_WILLNEED) != 0) {
        return 0;
      }
      return static_cast<int64_t>(clamped_length);
    }

    const auto cold_advice = mmap_cold_advice();
    if (cold_advice == MmapColdAdvice::invalidate) {
#if defined(MS_INVALIDATE) && defined(MS_ASYNC)
      if (msync(aligned_address, aligned_length, MS_INVALIDATE | MS_ASYNC) !=
          0) {
        return 0;
      }
      return static_cast<int64_t>(clamped_length);
#else
      return 0;
#endif
    }

    if (madvise(
            aligned_address,
            aligned_length,
            mmap_cold_madvise_value(cold_advice)) != 0) {
      return 0;
    }
    return static_cast<int64_t>(clamped_length);
  }

  std::mutex mutex_;
  std::vector<MmapTensorRegion> regions_;
  std::vector<MmapTensorRegion> layer_regions_;
};

std::optional<SafetensorsLoad> load_safetensors_mmap(
    const std::string& file,
    StreamOrDevice,
    const std::unordered_set<std::string>* excluded_keys = nullptr,
    std::optional<bool> tensor_buffers_override = std::nullopt) {
  const bool debug = mmap_debug_enabled();
  const bool tensor_buffers =
      tensor_buffers_override.value_or(mmap_tensor_buffers_enabled());
  auto log = [&](const char* message) {
    if (debug) {
      std::fprintf(
          stderr, "[mlx.safetensors.mmap] %s file=%s\n", message, file.c_str());
      std::fflush(stderr);
    }
  };
  int fd = open(file.c_str(), O_RDONLY | O_BINARY);
  if (fd < 0) {
    log("open-failed");
    return std::nullopt;
  }
  auto close_fd = [&]() {
    if (fd >= 0) {
      close(fd);
      fd = -1;
    }
  };
  struct stat st{};
  if (fstat(fd, &st) != 0 || st.st_size <= 0) {
    close_fd();
    log("stat-failed");
    return std::nullopt;
  }
  const auto file_size = static_cast<size_t>(st.st_size);
  void* raw = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
  if (raw == MAP_FAILED) {
    close_fd();
    log("mmap-failed");
    return std::nullopt;
  }
  if (!tensor_buffers) {
    close_fd();
  }

  auto unmap_on_failure = [&]() {
    if (raw && raw != MAP_FAILED) {
      munmap(raw, file_size);
      raw = nullptr;
    }
    close_fd();
  };

  if (file_size < 8) {
    unmap_on_failure();
    log("header-too-short");
    return std::nullopt;
  }

  const auto* base = static_cast<const char*>(raw);
  uint64_t json_header_length = 0;
  std::memcpy(&json_header_length, base, sizeof(json_header_length));
  constexpr uint64_t kMaxJsonHeaderLength = 100000000;
  if (json_header_length == 0 || json_header_length >= kMaxJsonHeaderLength ||
      8 + json_header_length > file_size) {
    unmap_on_failure();
    log("invalid-header-length");
    return std::nullopt;
  }

  json metadata;
  try {
    metadata = json::parse(base + 8, base + 8 + json_header_length);
  } catch (...) {
    unmap_on_failure();
    log("json-parse-failed");
    return std::nullopt;
  }
  if (!metadata.is_object()) {
    unmap_on_failure();
    log("metadata-not-object");
    return std::nullopt;
  }

  std::shared_ptr<MmapShard> shard;
  std::optional<array> base_array;
  if (!tensor_buffers) {
    shard = std::make_shared<MmapShard>(raw, file_size, file);
    raw = nullptr;
    auto buffer = allocator::make_buffer(shard->base, file_size);
    if (buffer.ptr() == nullptr) {
      log("make-buffer-failed");
      return std::nullopt;
    }
    shard->tracked_buffer_bytes = file_size;
    base_array.emplace(
        buffer, Shape{1}, uint8, [shard](allocator::Buffer buffer) {
          allocator::release(buffer);
        });
  }

  const size_t data_start = static_cast<size_t>(json_header_length) + 8;
  std::unordered_map<std::string, array> res;
  std::unordered_map<std::string, std::string> metadata_map;
  size_t realigned_tensor_count = 0;
  size_t realigned_tensor_bytes = 0;
  auto tensor_from_mmap = [&](const array& byte_base,
                              const Shape& shape,
                              Dtype type,
                              size_t byte_offset,
                              size_t byte_length) {
    if (byte_offset % size_of(type) == 0) {
      array tensor(
          allocator::Buffer(nullptr), shape, type, [](allocator::Buffer) {});
      tensor.copy_shared_buffer(
          byte_base,
          tensor.strides(),
          tensor.flags(),
          tensor.size(),
          static_cast<int64_t>(byte_offset / size_of(type)));
      return tensor;
    }

    // Safetensors only guarantees that each tensor's relative data offset is
    // valid; it does not require the absolute file offset (header + relative
    // offset) to be naturally aligned for the tensor dtype.  JANG affine
    // shards commonly interleave F16 metadata and U32 packed weights, so an
    // otherwise valid shard can contain U32 payloads at offsets congruent to
    // 2 mod 4.  Rejecting one such tensor used to discard the mmap path for
    // the entire multi-gigabyte shard and silently materialize every tensor
    // through the resident reader.
    //
    // Metal typed-buffer reads require natural alignment.  Copy only this
    // tensor into an aligned MLX buffer; all naturally aligned tensors in the
    // same shard remain zero-copy mmap-backed.  This bounds the unavoidable
    // repair to malformed payloads instead of materializing the entire shard.
    auto aligned = allocator::malloc(byte_length);
    if (aligned.ptr() == nullptr || aligned.raw_ptr() == nullptr) {
      throw std::runtime_error(
          "[safetensor] failed to allocate aligned buffer for unaligned mmap tensor");
    }
    std::memcpy(
        aligned.raw_ptr(),
        byte_base.data<uint8_t>() + byte_offset,
        byte_length);
    realigned_tensor_count += 1;
    realigned_tensor_bytes += byte_length;
    return array(aligned, shape, type);
  };
  for (const auto& item : metadata.items()) {
    if (item.key() == "__metadata__") {
      for (const auto& meta_item : item.value().items()) {
        metadata_map.insert({meta_item.key(), meta_item.value()});
      }
      continue;
    }
    if (excluded_keys && excluded_keys->count(item.key()) != 0) {
      continue;
    }

    const std::string& dtype = item.value().at("dtype");
    const Shape& shape = item.value().at("shape");
    const std::vector<size_t>& data_offsets = item.value().at("data_offsets");
    if (data_offsets.size() != 2 || data_offsets[1] < data_offsets[0]) {
      unmap_on_failure();
      log("invalid-data-offsets");
      return std::nullopt;
    }

    Dtype type = dtype_from_safetensor_str(dtype);
    const auto tensor_offset = data_start + data_offsets[0];
    const auto tensor_length = data_offsets[1] - data_offsets[0];
    if (tensor_offset > file_size ||
        tensor_length > file_size - tensor_offset) {
      log("tensor-out-of-bounds");
      return std::nullopt;
    }
    const bool tensor_is_mmap_backed = tensor_offset % size_of(type) == 0;
    std::optional<array> tensor;
    if (tensor_buffers) {
      const auto page = static_cast<size_t>(getpagesize());
      const auto aligned_start = tensor_offset & ~(page - 1);
      const auto aligned_end = tensor_offset + tensor_length;
      if (aligned_end <= aligned_start || aligned_end > file_size) {
        log("tensor-buffer-span-invalid");
        unmap_on_failure();
        return std::nullopt;
      }
      const auto span = aligned_end - aligned_start;
      void* tensor_raw = mmap(
          nullptr,
          span,
          PROT_READ,
          MAP_SHARED,
          fd,
          static_cast<off_t>(aligned_start));
      if (tensor_raw == MAP_FAILED) {
        log("tensor-buffer-mmap-failed");
        unmap_on_failure();
        return std::nullopt;
      }
      auto tensor_shard = std::make_shared<MmapShard>(tensor_raw, span, file);
      auto buffer = allocator::make_buffer(tensor_shard->base, span);
      if (buffer.ptr() == nullptr) {
        tensor_shard.reset();
        log("tensor-buffer-make-buffer-failed");
        unmap_on_failure();
        return std::nullopt;
      }
      tensor_shard->tracked_buffer_bytes = span;
      array tensor_base(
          buffer, Shape{1}, uint8, [tensor_shard](allocator::Buffer buffer) {
            allocator::release(buffer);
          });
      tensor = tensor_from_mmap(
          tensor_base,
          shape,
          type,
          tensor_offset - aligned_start,
          tensor_length);

      if (tensor_is_mmap_backed) {
        if (auto layer = match_layer_name(item.key())) {
          SafetensorsMmapRegistry::instance().register_layer_region(
              tensor_shard,
              *layer,
              tensor_offset - aligned_start,
              tensor_length);
        }

        if (auto routed = match_routed_name(item.key())) {
          if (routed->stacked && !shape.empty() && shape[0] > 1) {
            const auto experts = static_cast<size_t>(shape[0]);
            if (experts > 0 && tensor_length % experts == 0) {
              const auto per_expert = tensor_length / experts;
              for (size_t expert = 0; expert < experts; ++expert) {
                SafetensorsMmapRegistry::instance().register_region(
                    tensor_shard,
                    routed->layer,
                    static_cast<int32_t>(expert),
                    tensor_offset - aligned_start + expert * per_expert,
                    per_expert);
              }
            } else {
              SafetensorsMmapRegistry::instance().register_region(
                  tensor_shard,
                  routed->layer,
                  routed->expert,
                  tensor_offset - aligned_start,
                  tensor_length);
            }
          } else {
            SafetensorsMmapRegistry::instance().register_region(
                tensor_shard,
                routed->layer,
                routed->expert,
                tensor_offset - aligned_start,
                tensor_length);
          }
        }
      }
    } else {
      tensor = tensor_from_mmap(
          *base_array, shape, type, tensor_offset, tensor_length);

      if (tensor_is_mmap_backed) {
        if (auto layer = match_layer_name(item.key())) {
          SafetensorsMmapRegistry::instance().register_layer_region(
              shard, *layer, tensor_offset, tensor_length);
        }

        if (auto routed = match_routed_name(item.key())) {
          if (routed->stacked && !shape.empty() && shape[0] > 1) {
            const auto experts = static_cast<size_t>(shape[0]);
            if (experts > 0 && tensor_length % experts == 0) {
              const auto per_expert = tensor_length / experts;
              for (size_t expert = 0; expert < experts; ++expert) {
                SafetensorsMmapRegistry::instance().register_region(
                    shard,
                    routed->layer,
                    static_cast<int32_t>(expert),
                    tensor_offset + expert * per_expert,
                    per_expert);
              }
            } else {
              SafetensorsMmapRegistry::instance().register_region(
                  shard,
                  routed->layer,
                  routed->expert,
                  tensor_offset,
                  tensor_length);
            }
          } else {
            SafetensorsMmapRegistry::instance().register_region(
                shard,
                routed->layer,
                routed->expert,
                tensor_offset,
                tensor_length);
          }
        }
      }
    }
    res.insert({item.key(), std::move(*tensor)});
  }

  if (realigned_tensor_count > 0) {
    std::fprintf(
        stderr,
        "[mlx.safetensors.mmap] realigned %zu unaligned tensor(s) (%zu bytes) without shard fallback file=%s\n",
        realigned_tensor_count,
        realigned_tensor_bytes,
        file.c_str());
    std::fflush(stderr);
  }

  if (mmap_start_cold_enabled()) {
    SafetensorsMmapRegistry::instance().advise_routed(0, mmap_start_cold_pct());
  }

  if (tensor_buffers) {
    if (raw && raw != MAP_FAILED) {
      munmap(raw, file_size);
      raw = nullptr;
    }
    close_fd();
  }

  if (debug) {
    std::fprintf(
        stderr,
        "[mlx.safetensors.mmap] loaded file=%s bytes=%zu tensors=%zu\n",
        file.c_str(),
        file_size,
        res.size());
    std::fflush(stderr);
  }

  return SafetensorsLoad{std::move(res), std::move(metadata_map)};
}

} // namespace
#endif

int64_t safetensors_mmap_advise_routed(int32_t advice, int32_t cold_pct) {
#ifndef _WIN32
  return SafetensorsMmapRegistry::instance().advise_routed(advice, cold_pct);
#else
  return 0;
#endif
}

int64_t safetensors_mmap_advise_experts(
    int32_t advice,
    const int32_t* layers,
    const int32_t* experts,
    int64_t count) {
#ifndef _WIN32
  return SafetensorsMmapRegistry::instance().advise_experts(
      advice, layers, experts, count);
#else
  return 0;
#endif
}

int64_t safetensors_mmap_advise_layer(int32_t advice, int32_t layer) {
#ifndef _WIN32
  return SafetensorsMmapRegistry::instance().advise_layer(advice, layer);
#else
  return 0;
#endif
}

int64_t safetensors_mmap_tracked_buffer_bytes() {
#ifndef _WIN32
  return SafetensorsMmapRegistry::instance().tracked_buffer_bytes();
#else
  return 0;
#endif
}

static array mmap_file_region_impl(
    const std::string& file,
    uint64_t offset,
    size_t length,
    Shape shape,
    Dtype dtype,
    std::optional<int32_t> registered_layer) {
#ifndef _WIN32
  if (length == 0) {
    throw std::runtime_error("[mmap_file_region] length must be non-zero.");
  }
  size_t expected_length = size_of(dtype);
  for (auto dim : shape) {
    if (dim < 0) {
      throw std::runtime_error(
          "[mmap_file_region] shape contains negative dim.");
    }
    expected_length *= static_cast<size_t>(dim);
  }
  if (expected_length != length) {
    throw std::runtime_error("[mmap_file_region] shape byte count mismatch.");
  }

  int fd = open(file.c_str(), O_RDONLY | O_BINARY);
  if (fd < 0) {
    throw std::runtime_error("[mmap_file_region] open failed: " + file);
  }
  auto close_fd = [&]() {
    if (fd >= 0) {
      close(fd);
      fd = -1;
    }
  };

  struct stat st{};
  if (fstat(fd, &st) != 0 || st.st_size <= 0) {
    close_fd();
    throw std::runtime_error("[mmap_file_region] stat failed: " + file);
  }
  const auto file_size = static_cast<uint64_t>(st.st_size);
  if (offset > file_size || length > file_size - offset) {
    close_fd();
    throw std::runtime_error(
        "[mmap_file_region] region out of bounds: " + file);
  }

  const auto item_size = static_cast<uint64_t>(size_of(dtype));
  const auto page = static_cast<uint64_t>(getpagesize());
  const auto aligned_start = offset & ~(page - 1);
  const auto offset_delta = offset - aligned_start;
  if (offset_delta % item_size != 0) {
    close_fd();
    throw std::runtime_error("[mmap_file_region] dtype-unaligned offset.");
  }
  const auto span = offset_delta + static_cast<uint64_t>(length);
  if (span > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    close_fd();
    throw std::runtime_error("[mmap_file_region] span too large.");
  }

  void* raw = mmap(
      nullptr,
      static_cast<size_t>(span),
      PROT_READ,
      MAP_SHARED,
      fd,
      static_cast<off_t>(aligned_start));
  close_fd();
  if (raw == MAP_FAILED) {
    throw std::runtime_error("[mmap_file_region] mmap failed: " + file);
  }

  auto shard =
      std::make_shared<MmapShard>(raw, static_cast<size_t>(span), file);
  auto buffer = allocator::make_buffer(shard->base, static_cast<size_t>(span));
  if (buffer.ptr() == nullptr) {
    throw std::runtime_error("[mmap_file_region] make_buffer failed: " + file);
  }
  shard->tracked_buffer_bytes = static_cast<size_t>(span);
  array base(
      buffer,
      Shape{static_cast<ShapeElem>(span)},
      uint8,
      [shard](allocator::Buffer buffer) { allocator::release(buffer); });
  array tensor(
      allocator::Buffer(nullptr), shape, dtype, [](allocator::Buffer) {});
  tensor.copy_shared_buffer(
      base,
      tensor.strides(),
      tensor.flags(),
      tensor.size(),
      static_cast<int64_t>(offset_delta / item_size));
  if (registered_layer) {
    auto& registry = SafetensorsMmapRegistry::instance();
    registry.register_layer_region(
        shard, *registered_layer, static_cast<size_t>(offset_delta), length);
    const auto experts = static_cast<size_t>(shape[0]);
    const auto per_expert = length / experts;
    for (size_t expert = 0; expert < experts; ++expert) {
      registry.register_region(
          shard,
          *registered_layer,
          static_cast<int32_t>(expert),
          static_cast<size_t>(offset_delta) + expert * per_expert,
          per_expert);
    }
  }
  return tensor;
#else
  throw std::runtime_error("[mmap_file_region] unsupported on Windows.");
#endif
}

array mmap_file_region(
    const std::string& file,
    uint64_t offset,
    size_t length,
    Shape shape,
    Dtype dtype) {
  return mmap_file_region_impl(
      file, offset, length, std::move(shape), dtype, std::nullopt);
}

array mmap_file_region_named(
    const std::string& file,
    uint64_t offset,
    size_t length,
    Shape shape,
    Dtype dtype,
    const std::string& tensor_name) {
  // Admit routed banks and one-expert dense down banks. The old loader's
  // name matching and unnamed API are unchanged.
  // Routed banks may sit under a VLM `language_model.` wrapper (MiniCPM-V 4.7
  // 35B-A3B JANGH2, 2026-10-12). The layer index stays capture group 1.
  static const std::regex canonical(
      R"(^(?:language_model\.)?model\.layers\.(0|[1-9][0-9]*)\.(?:mlp|feed_forward)\.switch_mlp\.(gate|up|down)_proj\.tq2_(packed|scales)$)");
  // Dense one-expert banks: K2 down projections
  // (`model.layers.L.mlp.down_proj`) and the Qwen3.5-family dense JANGH MLP
  // (`language_model.model.layers.L.mlp.{gate,up,down}_proj`, Qwen3.8-27B
  // JANGH2, 2026-10-06). The layer index stays capture group 1 for the
  // JangPress layer registry.
  static const std::regex dense_canonical(
      R"(^(?:language_model\.)?model\.layers\.(0|[1-9][0-9]*)\.mlp\.(gate|up|down)_proj\.tq2_(packed|scales)$)");
  // Nemotron-H latent MoE (Nemotron 3.5 Bolt JANGH2): gate-less relu2 routed
  // banks at `backbone.layers.L.mixer.switch_mlp.{fc1,fc2}`. The layer index
  // stays capture group 1 and the packed/scales tag group 3.
  static const std::regex relu2_canonical(
      R"(^backbone\.layers\.(0|[1-9][0-9]*)\.mixer\.switch_mlp\.(fc1|fc2)\.tq2_(packed|scales)$)");
  std::smatch match;
  bool dense = false;
  if (!std::regex_match(tensor_name, match, canonical) &&
      !std::regex_match(tensor_name, match, relu2_canonical)) {
    if (!std::regex_match(tensor_name, match, dense_canonical)) {
      throw std::invalid_argument(
          "[mmap_file_region_named] invalid canonical bank name.");
    }
    dense = true;
  }
  const auto layer_value = std::stoull(match[1].str());
  if (layer_value >
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    throw std::invalid_argument(
        "[mmap_file_region_named] layer index overflow.");
  }
  const bool packed = match[3].str() == "packed";
  if (shape.size() != (packed ? 3 : 2) ||
      dtype != (packed ? uint32 : float16) || (dense && shape[0] != 1)) {
    throw std::invalid_argument(
        "[mmap_file_region_named] bank rank/dtype mismatch.");
  }
  size_t expected = size_of(dtype);
  for (auto dim : shape) {
    if (dim <= 0 ||
        expected >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
      throw std::invalid_argument(
          "[mmap_file_region_named] invalid bank dimensions.");
    }
    expected *= static_cast<size_t>(dim);
  }
  if (expected != length || length % static_cast<size_t>(shape[0]) != 0) {
    throw std::invalid_argument(
        "[mmap_file_region_named] bank byte count mismatch.");
  }
#ifndef _WIN32
  const auto delta = offset % static_cast<uint64_t>(getpagesize());
  if (length >
      static_cast<uint64_t>(std::numeric_limits<ShapeElem>::max()) - delta) {
    throw std::invalid_argument(
        "[mmap_file_region_named] bank mapped span overflow.");
  }
#endif
  return mmap_file_region_impl(
      file,
      offset,
      length,
      std::move(shape),
      dtype,
      static_cast<int32_t>(layer_value));
}

/** Load array from reader in safetensor format */
SafetensorsLoad load_safetensors(
    std::shared_ptr<io::Reader> in_stream,
    StreamOrDevice s,
    const std::unordered_set<std::string>* excluded_keys) {
  ////////////////////////////////////////////////////////
  // Open and check file
  if (!in_stream->good() || !in_stream->is_open()) {
    std::ostringstream msg;
    msg << "[load_safetensors] Failed to open " << in_stream->label();
    throw std::runtime_error(msg.str());
  }

  auto stream = cu::is_available() ? to_stream(s) : to_stream(s, Device::cpu);

  uint64_t jsonHeaderLength = 0;
  // This is the same limit as in the original Rust Safetensors code.
  constexpr uint64_t kMaxJsonHeaderLength = 100000000;
  in_stream->read(reinterpret_cast<char*>(&jsonHeaderLength), 8);
  if (jsonHeaderLength <= 0 || jsonHeaderLength >= kMaxJsonHeaderLength) {
    std::ostringstream msg;
    msg << "[load_safetensors] Invalid json header length "
        << in_stream->label();
    throw std::runtime_error(msg.str());
  }

  // Determine file size to be able to validate that reads are within the
  // bounds of the file (at least at creation time)
  in_stream->seek(0, std::ios_base::end);
  size_t file_size = in_stream->tell();
  in_stream->seek(8, std::ios_base::beg);

  // Load the json metadata
  if (file_size < jsonHeaderLength + 8) {
    std::ostringstream msg;
    msg << "[load_safetensors] The JSON header is " << jsonHeaderLength
        << " bytes long but the file is only " << file_size << " bytes. "
        << "Perhaps an incomplete download or corrupt file?";
    throw std::runtime_error(msg.str());
  }
  auto rawJson = std::make_unique<char[]>(jsonHeaderLength);
  in_stream->read(rawJson.get(), jsonHeaderLength);
  auto metadata = json::parse(rawJson.get(), rawJson.get() + jsonHeaderLength);
  // Should always be an object on the top-level
  if (!metadata.is_object()) {
    std::ostringstream msg;
    msg << "[load_safetensors] Invalid json metadata " << in_stream->label();
    throw std::runtime_error(msg.str());
  }
  size_t offset = jsonHeaderLength + 8;

  // Load the arrays using metadata
  std::unordered_map<std::string, array> res;
  std::unordered_map<std::string, std::string> metadata_map;
  for (const auto& item : metadata.items()) {
    if (item.key() == "__metadata__") {
      for (const auto& meta_item : item.value().items()) {
        metadata_map.insert({meta_item.key(), meta_item.value()});
      }
      continue;
    }
    if (excluded_keys && excluded_keys->count(item.key()) != 0) {
      continue;
    }
    const std::string& dtype = item.value().at("dtype");
    const Shape& shape = item.value().at("shape");
    const std::vector<size_t>& data_offsets = item.value().at("data_offsets");
    Dtype type = dtype_from_safetensor_str(dtype);
    if (data_offsets.size() != 2) {
      std::ostringstream msg;
      msg << "[load_safetensors] Tensor '" << item.key()
          << "' data_offsets must have exactly 2 entries but has "
          << data_offsets.size();
      throw std::runtime_error(msg.str());
    }
    {
      size_t expected_nbytes = type.size();
      for (auto dim : shape) {
        expected_nbytes *= static_cast<size_t>(dim);
      }
      if (data_offsets[1] < data_offsets[0] ||
          data_offsets[1] - data_offsets[0] != expected_nbytes) {
        std::ostringstream msg;
        msg << "[load_safetensors] Tensor '" << item.key()
            << "' invalid data offsets (" << data_offsets[0] << ", "
            << data_offsets[1] << "). Expecting " << expected_nbytes
            << " bytes.";
        throw std::runtime_error(msg.str());
      }
    }
    if (offset + data_offsets[1] > file_size) {
      std::ostringstream msg;
      msg << "[load_safetensors] Tensor '" << item.key()
          << "' invalid data offsets (" << data_offsets[0] << ", "
          << data_offsets[1] << ") exceeding the size of the file. "
          << "Perhaps an incomplete download or corrupt file?";
      throw std::runtime_error(msg.str());
    }
    res.insert(
        {item.key(),
         array(
             shape,
             type,
             std::make_shared<Load>(
                 stream, in_stream, offset + data_offsets.at(0), false),
             std::vector<array>{})});
  }
  return {res, metadata_map};
}

SafetensorsLoad load_safetensors(
    std::shared_ptr<io::Reader> in_stream,
    StreamOrDevice s) {
  return load_safetensors(std::move(in_stream), s, nullptr);
}

SafetensorsLoad load_safetensors(const std::string& file, StreamOrDevice s) {
  return load_safetensors(std::make_shared<io::ParallelFileReader>(file), s);
}

SafetensorsLoad load_safetensors_excluding(
    const std::string& file,
    const std::unordered_set<std::string>& excluded_keys,
    StreamOrDevice s) {
  return load_safetensors(
      std::make_shared<io::ParallelFileReader>(file), s, &excluded_keys);
}

SafetensorsLoad load_safetensors_excluding(
    const std::string& file,
    const std::unordered_set<std::string>& excluded_keys,
    bool /* exact_tensor_buffers */,
    StreamOrDevice s) {
  return load_safetensors(
      std::make_shared<io::ParallelFileReader>(file), s, &excluded_keys);
}

void save_safetensors(
    std::shared_ptr<io::Writer> out_stream,
    std::unordered_map<std::string, array> a,
    std::unordered_map<std::string, std::string> metadata /* = {} */) {
  {
    std::vector<array> to_eval;
    to_eval.reserve(a.size());
    for (auto& p : a) {
      p.second = contiguous(p.second);
      to_eval.push_back(p.second);
    }
    eval(std::move(to_eval));
  }

  // Open only after the inputs are evaluated: opening a FileWriter
  // truncates the file and lazy inputs may still read from it.
  out_stream->open();

  ////////////////////////////////////////////////////////
  // Check file
  if (!out_stream->good() || !out_stream->is_open()) {
    std::ostringstream msg;
    msg << "[save_safetensors] Failed to open " << out_stream->label();
    throw std::runtime_error(msg.str());
  }

  ////////////////////////////////////////////////////////
  // Check array map
  json parent;
  json _metadata;
  for (auto& [key, value] : metadata) {
    _metadata[key] = value;
  }
  parent["__metadata__"] = _metadata;

  size_t offset = 0;
  for (auto& [key, arr] : a) {
    json child;
    child["dtype"] = dtype_to_safetensor_str(arr.dtype());
    child["shape"] = arr.shape();
    child["data_offsets"] = std::vector<size_t>{offset, offset + arr.nbytes()};
    parent[key] = child;
    offset += arr.nbytes();
  }

  auto header = parent.dump();
  uint64_t header_len = header.length();
  out_stream->write(reinterpret_cast<char*>(&header_len), 8);
  out_stream->write(header.c_str(), header_len);
  for (auto& [key, arr] : a) {
    // An empty tensor contributes a zero length span and has no data pointer
    // worth asking for.
    if (arr.nbytes() > 0) {
      out_stream->write(arr.data<char>(), arr.nbytes());
    }
  }
}

void save_safetensors(
    std::string file,
    std::unordered_map<std::string, array> a,
    std::unordered_map<std::string, std::string> metadata /* = {} */) {
  // Add .safetensors to file name if it is not there
  if (file.length() < 12 ||
      file.substr(file.length() - 12, 12) != ".safetensors")
    file += ".safetensors";

  // Serialize array
  save_safetensors(
      std::make_shared<io::FileWriter>(std::move(file)), a, metadata);
}

} // namespace mlx::core
