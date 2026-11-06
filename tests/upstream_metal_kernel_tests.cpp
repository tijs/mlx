#include "doctest/doctest.h"
#include "mlx/fast.h"
#include "mlx/mlx.h"
#include <cmath>

using namespace mlx::core;

namespace {
void check_relative_rms(const array& actual, const array& expected, float tolerance) {
  auto actual_cpu = astype(actual, float32, Device::cpu);
  auto reference_cpu = astype(expected, float32, Device::cpu);
  auto error = sqrt(mean(square(subtract(actual_cpu, reference_cpu, Device::cpu), Device::cpu), Device::cpu), Device::cpu);
  auto scale = maximum(sqrt(mean(square(reference_cpu, Device::cpu), Device::cpu), Device::cpu), array(1e-5f), Device::cpu);
  auto relative = divide(error, scale, Device::cpu);
  eval(relative);
  CHECK(relative.item<float>() < tolerance);
}
}

TEST_CASE("test upstream Metal thin GEMM and quantized row boundaries") {
  if (!is_available(Device::gpu)) {
    return;
  }
  for (auto dtype : {float16, bfloat16}) {
    const float tolerance = dtype == bfloat16 ? 0.015f : 0.003f;
    auto x = astype(reshape(sin(divide(arange(65 * 2560, float32, Device::cpu), array(31.0f), Device::cpu), Device::cpu), {65, 2560}, Device::cpu), dtype, Device::gpu);
    auto w = astype(reshape(cos(divide(arange(1024 * 2560, float32, Device::cpu), array(17.0f), Device::cpu), Device::cpu), {1024, 2560}, Device::cpu), dtype, Device::gpu);
    for (int rows : {6, 16, 33, 65}) {
      CAPTURE(dtype);
      CAPTURE(rows);
      auto input = slice(x, {0, 0}, {rows, 2560});
      for (int bits : {1, 2, 4, 6, 8}) {
        CAPTURE(bits);
        auto quantized = quantize(w, 64, bits, "affine", std::nullopt, Device::gpu);
        auto decoded = dequantize(quantized[0], quantized[1], quantized[2], 64, bits, "affine", std::nullopt, dtype, Device::cpu);
        auto expected = matmul(astype(input, float32, Device::cpu), transpose(astype(decoded, float32, Device::cpu), Device::cpu), Device::cpu);
        auto actual = quantized_matmul(input, quantized[0], quantized[1], quantized[2], true, 64, bits, "affine", Device::gpu);
        check_relative_rms(actual, expected, tolerance);
      }
    }
    auto a = astype(reshape(sin(arange(257 * 255, float32, Device::cpu), Device::cpu), {257, 255}, Device::cpu), dtype, Device::gpu);
    auto b = astype(reshape(cos(arange(255 * 17, float32, Device::cpu), Device::cpu), {255, 17}, Device::cpu), dtype, Device::gpu);
    check_relative_rms(matmul(a, b, Device::gpu), matmul(astype(a, float32, Device::cpu), astype(b, float32, Device::cpu), Device::cpu), tolerance);
  }
}

TEST_CASE("test upstream Metal two-pass attention preserves sparse masked keys") {
  if (!is_available(Device::gpu)) {
    return;
  }
  constexpr int length = 65536, width = 128;
  auto query = ones({1, 8, 1, width}, float16, Device::gpu);
  auto keys = zeros({1, 2, length, width}, float16, Device::gpu);
  auto values = reshape(astype(divide(remainder(arange(2 * length * width, int32, Device::cpu), array(17), Device::cpu), array(17.0f), Device::cpu), float16, Device::gpu), {1, 2, length, width});
  auto mask = reshape(equal(remainder(arange(length, int32, Device::gpu), array(7), Device::gpu), array(0), Device::gpu), {1, 1, 1, length});
  auto masked_scores = where(mask, zeros({1, 8, 1, length}, float32, Device::cpu), full({1, 8, 1, length}, -INFINITY, float32, Device::cpu), Device::cpu);
  auto repeated_values = repeat(astype(values, float32, Device::cpu), 4, 1, Device::cpu);
  auto expected = matmul(softmax(masked_scores, -1, true, Device::cpu), repeated_values, Device::cpu);
  auto actual = fast::scaled_dot_product_attention(query, keys, values, 1.0f, "", mask);
  check_relative_rms(actual, expected, 0.003f);
}

TEST_CASE("test upstream Metal FP4 half and bfloat fallback numerical parity") {
  if (!is_available(Device::gpu)) {
    return;
  }
  for (auto dtype : {float16, bfloat16}) {
    for (const auto& mode : {std::string("mxfp4"), std::string("nvfp4")}) {
      const int group_size = mode == "nvfp4" ? 16 : 32;
      auto w = astype(reshape(sin(arange(2048 * 2560, float32, Device::cpu), Device::cpu), {2048, 2560}, Device::cpu), dtype, Device::gpu);
      auto quantized = quantize(w, group_size, 4, mode, std::nullopt, Device::gpu);
      auto decoded = dequantize(quantized[0], quantized[1], std::nullopt, group_size, 4, mode, std::nullopt, float32, Device::cpu);
      for (int rows : {6, 16, 33, 65}) {
        CAPTURE(dtype);
        CAPTURE(mode);
        CAPTURE(rows);
        auto input = astype(reshape(cos(arange(rows * 2560, float32, Device::cpu), Device::cpu), {rows, 2560}, Device::cpu), dtype, Device::gpu);
        auto expected = matmul(astype(input, float32, Device::cpu), transpose(decoded, Device::cpu), Device::cpu);
        auto actual = quantized_matmul(input, quantized[0], quantized[1], std::nullopt, true, group_size, 4, mode, Device::gpu);
        check_relative_rms(actual, expected, dtype == bfloat16 ? 0.015f : 0.003f);
      }
    }
  }
}

TEST_CASE("test one-bit affine endpoints CPU GPU and small-K dispatch") {
  REQUIRE(is_available(Device::gpu));
  for (auto dtype : {float32, float16, bfloat16}) {
    for (int width : {64, 128}) {
      for (int group : {32, 64, 128}) {
        if (width % group != 0) {
          continue;
        }
        for (int endpoints = 0; endpoints < 4; ++endpoints) {
          CAPTURE(dtype);
          CAPTURE(width);
          CAPTURE(group);
          CAPTURE(endpoints);
          const float low[] = {-0.5f, 1.0f, -2.0f, 0.0f};
          const float high[] = {0.5f, 2.0f, -1.0f, 0.0f};
          auto alternating = reshape(remainder(arange(16 * width, int32, Device::cpu), array(2), Device::cpu), {16, width});
          auto weights = astype(where(alternating, array(high[endpoints]), array(low[endpoints]), Device::cpu), dtype, Device::cpu);
          auto input = ones({1, width}, dtype, Device::cpu);
          auto expected = matmul(astype(input, float32, Device::cpu), transpose(astype(weights, float32, Device::cpu), Device::cpu), Device::cpu);
          for (auto device : {Device::cpu, Device::gpu}) {
            CAPTURE(device);
            auto quantized = quantize(astype(weights, dtype, device), group, 1, "affine", std::nullopt, device);
            auto decoded = dequantize(quantized[0], quantized[1], quantized[2], group, 1, "affine", std::nullopt, dtype, Device::cpu);
            INFO("scales=" << quantized[1] << " biases=" << quantized[2]);
            INFO("decoded=" << decoded << " weights=" << weights);
            REQUIRE(array_equal(decoded, weights, false, Device::cpu).item<bool>());
            auto result = quantized_matmul(astype(input, dtype, device), quantized[0], quantized[1], quantized[2], true, group, 1, "affine", device);
            check_relative_rms(result, expected, dtype == bfloat16 ? 0.015f : 0.003f);
          }
        }
      }
    }
  }
}
