#include <metal_stdlib>
using namespace metal;
kernel void add_one(device float* values [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    values[i] += 1.0f;
}
