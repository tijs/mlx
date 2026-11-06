// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"

using namespace mlx::core;

TEST_CASE("test empty gather output constructs without dividing by zero") {
  for (auto device : {Device::cpu, Device::gpu}) {
    if (!is_available(device)) {
      continue;
    }
    auto stream = new_stream(device);
    auto output = gather_qmm(
        ones({16, 1, 512}, bfloat16, stream),
        zeros({4, 0, 64}, uint32, stream),
        zeros({4, 0, 8}, float16, stream),
        zeros({4, 0, 8}, float16, stream),
        std::nullopt,
        zeros({16}, uint32, stream),
        true,
        64,
        4,
        "affine",
        std::nullopt,
        true,
        stream);
    CHECK(output.shape() == Shape{16, 1, 0});
    CHECK(output.dtype() == float32);
    CHECK(output.primitive().stream() == stream);
  }
}

TEST_CASE("test mixed affine gather storage follows the selected consumer") {
  for (auto device : {Device::cpu, Device::gpu}) {
    if (!is_available(device)) {
      continue;
    }
    auto stream = new_stream(device);
    for (int bits : {4, 8}) {
      for (int experts : {2, 4}) {
        for (int rows : {1, 15, 16}) {
          for (bool sorted : {false, true}) {
            CAPTURE(device);
            CAPTURE(bits);
            CAPTURE(experts);
            CAPTURE(rows);
            CAPTURE(sorted);
            auto x = ones({rows, 1, 512}, bfloat16, stream);
            auto w = zeros({experts, 8, 512 * bits / 32}, uint32, stream);
            auto scales = zeros({experts, 8, 8}, float16, stream);
            // Packed zeros plus exact biases expose half/BF16 reinterpretation
            // independently of matmul reduction order.
            auto bias = broadcast_to(
                reshape(
                    multiply(
                        add(arange(experts, float32, stream), array(1.0f)),
                        array(0.03125f),
                        stream),
                    {experts, 1, 1},
                    stream),
                {experts, 8, 8},
                stream);
            bias = astype(bias, float16, stream);
            auto ids = floor_divide(
                multiply(arange(rows, uint32, stream), array(experts, uint32)),
                array(rows, uint32),
                stream);
            auto output = gather_qmm(
                x,
                w,
                scales,
                bias,
                std::nullopt,
                ids,
                true,
                64,
                bits,
                "affine",
                std::nullopt,
                sorted,
                stream);
            const bool matrix = sorted && rows >= 16 && rows / experts >= 4;
            const bool mixed_vector =
                device == Device::gpu && metal::is_available() && !matrix;
            CHECK(output.dtype() == (mixed_vector ? bfloat16 : float32));
            CHECK(output.primitive().stream() == stream);
            auto expected = broadcast_to(
                reshape(
                    multiply(
                        add(astype(ids, float32, stream), array(1.0f)),
                        array(16.0f),
                        stream),
                    {rows, 1, 1},
                    stream),
                {rows, 1, 8},
                stream);
            CHECK(array_equal(
                      astype(output, float32, stream), expected, false, stream)
                      .item<bool>());
            synchronize(stream);
          }
        }
      }
    }
  }
}

TEST_CASE("test explicit gather lhs indices preserve mixed vector dispatch") {
  if (!is_available(Device::gpu) || !metal::is_available()) {
    return;
  }
  auto stream = new_stream(Device::gpu);
  auto x = ones({16, 1, 512}, bfloat16, stream);
  auto w = zeros({2, 8, 64}, uint32, stream);
  auto scales = zeros({2, 8, 8}, float16, stream);
  auto biases = full({2, 8, 8}, 0.03125f, float16, stream);
  auto lhs = arange(16, uint32, stream);
  auto rhs = zeros({16}, uint32, stream);
  auto output = gather_qmm(
      x,
      w,
      scales,
      biases,
      lhs,
      rhs,
      true,
      64,
      4,
      "affine",
      std::nullopt,
      true,
      stream);
  CHECK(output.dtype() == bfloat16);
  CHECK(array_equal(
            output, full({16, 1, 8}, 16.0f, bfloat16, stream), false, stream)
            .item<bool>());
  synchronize(stream);
}

TEST_CASE("test affine quantized matmul casts preserve explicit stream") {
  for (auto device : {Device::cpu, Device::gpu}) {
    if (!is_available(device)) {
      continue;
    }
    auto stream = new_stream(device);
    REQUIRE_FALSE(stream == default_stream(device));
    for (int bits : {2, 3, 4, 5, 6, 8}) {
      for (int group_size : {32, 64}) {
        for (int rows : {1, 3}) {
          CAPTURE(device);
          CAPTURE(bits);
          CAPTURE(group_size);
          CAPTURE(rows);
          // Seven output rows exclude the mixed BF16/F16 QMV fast path.
          // Both activations and affine metadata must promote to F32 here.
          auto weights = reshape(
              sin(arange(7 * 512, float32, stream), stream), {7, 512}, stream);
          auto q = quantize(
              astype(weights, float16, stream),
              group_size,
              bits,
              "affine",
              std::nullopt,
              stream);
          auto x = astype(
              reshape(
                  cos(arange(rows * 512, float32, stream), stream),
                  {rows, 512},
                  stream),
              bfloat16,
              stream);
          eval(x, q[0], q[1], q[2]);
          auto y = quantized_matmul(
              x, q[0], q[1], q[2], true, group_size, bits, "affine", stream);
          REQUIRE(y.has_primitive());
          CHECK(y.primitive().stream() == stream);
          CHECK(y.dtype() == float32);
          REQUIRE(y.inputs().size() == 4);
          for (int input : {0, 2, 3}) {
            REQUIRE(y.inputs()[input].has_primitive());
            CHECK(y.inputs()[input].dtype() == float32);
            CHECK(y.inputs()[input].primitive().stream() == stream);
          }
          // Scheduling must not change arithmetic: compare against explicitly
          // promoted inputs on the requested stream, with exact equality.
          auto reference = quantized_matmul(
              astype(x, float32, stream),
              q[0],
              astype(q[1], float32, stream),
              astype(q[2], float32, stream),
              true,
              group_size,
              bits,
              "affine",
              stream);
          CHECK(array_equal(y, reference, false, stream).item<bool>());
          synchronize(stream);
        }
      }
    }
  }
}
