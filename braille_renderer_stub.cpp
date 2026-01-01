// CPU stub for CUDA renderer functions (used when building without CUDA)
// Build with: make         (CPU-only, uses this stub)
// Build with: make CUDA=true (uses actual CUDA implementation)

#include <cstdint>
#include <cstring>

extern "C" void cuda_render_braille(
    const uint8_t* frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int w_scaled,
    int h_scaled,
    int cells_x,
    int cells_y,
    uint8_t detail_level,
    uint8_t threshold_steps,
    uint8_t* patterns,
    uint8_t* fg_colors,
    uint8_t* bg_colors) {
    // Stub: zero out patterns, white fg, black bg
    // Server will use CPU rendering path when render_device != cuda
    memset(patterns, 0, cells_x * cells_y);
    memset(fg_colors, 255, cells_x * cells_y * 3);
    memset(bg_colors, 0, cells_x * cells_y * 3);
}

extern "C" void cuda_render_hybrid(
    const uint8_t* frame,
    uint32_t fw, uint32_t fh, uint32_t stride,
    int w_scaled, int h_scaled,
    int cells_x, int cells_y,
    uint8_t detail, uint8_t threshold_steps,
    uint8_t* modes,
    uint8_t* patterns,
    uint8_t* fg_colors,
    uint8_t* bg_colors) {
    // Stub: zero out everything
    memset(modes, 0, cells_x * cells_y);
    memset(patterns, 0, cells_x * cells_y);
    memset(fg_colors, 255, cells_x * cells_y * 3);
    memset(bg_colors, 0, cells_x * cells_y * 3);
}


extern "C" void render_braille_cuda(
    const uint8_t* frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int w_scaled,
    int h_scaled,
    int cells_x,
    int cells_y,
    uint8_t detail_level,
    uint8_t threshold_steps,
    uint8_t* patterns,
    uint8_t* fg_colors,
    uint8_t* bg_colors) {
    // Stub: zero out patterns, white fg, black bg
    // Server will use CPU rendering path when render_device != cuda
    memset(patterns, 0, cells_x * cells_y);
    memset(fg_colors, 255, cells_x * cells_y * 3);
    memset(bg_colors, 0, cells_x * cells_y * 3);
}

extern "C" void render_hybrid_cuda(
    const uint8_t* frame,
    uint32_t fw, uint32_t fh, uint32_t stride,
    int w_scaled, int h_scaled,
    int cells_x, int cells_y,
    uint8_t detail, uint8_t threshold_steps,
    uint8_t* modes,
    uint8_t* patterns,
    uint8_t* fg_colors,
    uint8_t* bg_colors) {
    // Stub: zero out everything
    memset(modes, 0, cells_x * cells_y);
    memset(patterns, 0, cells_x * cells_y);
    memset(fg_colors, 255, cells_x * cells_y * 3);
    memset(bg_colors, 0, cells_x * cells_y * 3);
}
