#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
constant int ITERS [[function_constant(0)]];
// Both operands as device tensors (16 x 512 and 32 x 512, cache-resident), K from the tensors: 16x32x512 a call.
template <typename A, typename B, typename C>
kernel void bench(device float* out [[buffer(0)]], device uchar* raw [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
  constexpr auto shape = matmul2d_descriptor(16, 32, static_cast<int>(dynamic_extent), false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<shape, execution_simdgroup> op;
  tensor<device A, extents<int32_t, 512, 16>, tensor_inline> ta((metal::__tensor_detail::__tensor_data_handle_t<device A>)raw, extents<int32_t, 512, 16>());
  tensor<device B, extents<int32_t, 512, 32>, tensor_inline> tb((metal::__tensor_detail::__tensor_data_handle_t<device B>)(raw + 65536), extents<int32_t, 512, 32>());
  using AT = metal::remove_addrspace_t<decltype(ta)>;
  using BT = metal::remove_addrspace_t<decltype(tb)>;
  auto acc = op.template get_destination_cooperative_tensor<AT, BT, C>();
  for (ushort i = 0; i < acc.get_capacity(); i++) acc[i] = C(0);
  for (int it = 0; it < ITERS; it++) op.run(ta, tb, acc);
  float s = 0;
  for (ushort i = 0; i < acc.get_capacity(); i++) s += float(acc[i]);
  out[gid] = s;
}
template [[host_name("bf16_bf16_f32")]] kernel void bench<bfloat, bfloat, float>(device float*, device uchar*, uint);
template [[host_name("i8_i8_i32")]] kernel void bench<int8_t, int8_t, int32_t>(device float*, device uchar*, uint);
template [[host_name("bf16_i8_f32")]] kernel void bench<bfloat, int8_t, float>(device float*, device uchar*, uint);
template [[host_name("bf16_i4_f32")]] kernel void bench<bfloat, int4b_format, float>(device float*, device uchar*, uint);
template [[host_name("i8_i4_i32")]] kernel void bench<int8_t, int4b_format, int32_t>(device float*, device uchar*, uint);
