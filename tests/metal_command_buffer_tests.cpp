#include "doctest/doctest.h"

#include <tuple>

#include "mlx/backend/metal/device.h"
#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test Metal command buffer budget uses storage bytes") {
  const auto stream = new_stream(Device::gpu);
  auto& encoder = metal::get_command_encoder(stream);
  const auto max_mb =
      std::get<1>(metal::device(stream.device).get_max_ops_mb_per_buffer());
  CAPTURE(max_mb);
  REQUIRE(max_mb >= 0);
  REQUIRE(max_mb <= 256);
  const size_t threshold = (static_cast<size_t>(max_mb) + 1) << 20;

  for (const auto dtype : {float32, float16, bfloat16, uint8}) {
    CAPTURE(dtype);
    const auto count = threshold / size_of(dtype) + 1;
    auto input = arange(static_cast<double>(count), dtype, stream);
    eval(input);
    synchronize(stream);
    REQUIRE(input.data_size() == count);
    CHECK_FALSE(encoder.needs_commit());
    encoder.set_input_array(input, 0);
    CHECK(encoder.needs_commit());
    encoder.synchronize();
  }
}

TEST_CASE("test Metal command buffer budget counts broadcast storage once") {
  const auto stream = new_stream(Device::gpu);
  auto& encoder = metal::get_command_encoder(stream);
  const auto max_mb =
      std::get<1>(metal::device(stream.device).get_max_ops_mb_per_buffer());
  REQUIRE(max_mb >= 1);
  REQUIRE(max_mb <= 256);
  const auto count = ((max_mb + 1) << 20) / sizeof(float) + 1;
  auto input = broadcast_to(
      arange(1, float32, stream), {static_cast<int>(count)}, stream);
  eval(input);
  synchronize(stream);
  REQUIRE(input.data_size() == 1);
  REQUIRE(input.nbytes() > (static_cast<size_t>(max_mb) << 20));
  encoder.set_input_array(input, 0);
  CHECK_FALSE(encoder.needs_commit());
  encoder.synchronize();
}

TEST_CASE("test Metal command buffer budget deduplicates aliased inputs") {
  const auto stream = new_stream(Device::gpu);
  auto& encoder = metal::get_command_encoder(stream);
  const auto max_mb =
      std::get<1>(metal::device(stream.device).get_max_ops_mb_per_buffer());
  REQUIRE(max_mb >= 1);
  REQUIRE(max_mb <= 256);
  const auto count = ((max_mb + 1) << 20) / (2 * sizeof(float));
  auto input = arange(static_cast<double>(count), float32, stream);
  auto alias = reshape(input, {static_cast<int>(count), 1}, stream);
  eval(input, alias);
  synchronize(stream);
  REQUIRE(input.buffer().ptr() == alias.buffer().ptr());
  encoder.set_input_array(input, 0);
  encoder.set_input_array(alias, 1);
  CHECK_FALSE(encoder.needs_commit());
  encoder.synchronize();
}
