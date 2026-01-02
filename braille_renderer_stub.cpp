// stub for CUDA renderer (used when building without CUDA)
// make         -> uses this stub
// make CUDA=1  -> uses actual CUDA implementation

#include <cstdint>
#include <cstring>

extern "C" void render_braille_cuda(
    const uint8_t* frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int w_scaled,
    int h_scaled,
    int cells_x,
    int cells_y,
    double rotation_angle,
    uint8_t pixel_format,
    uint8_t detail_level,
    uint8_t threshold_steps,
    uint8_t* patterns,
    uint8_t* fg_colors,
    uint8_t* bg_colors) {
    (void)frame_data; (void)frame_width; (void)frame_height; (void)frame_stride;
    (void)w_scaled; (void)h_scaled; (void)rotation_angle; (void)pixel_format;
    (void)detail_level; (void)threshold_steps;
    memset(patterns, 0, cells_x * cells_y);
    memset(fg_colors, 255, cells_x * cells_y * 3);
    memset(bg_colors, 0, cells_x * cells_y * 3);
}

extern "C" void render_hybrid_cuda(
    const uint8_t* frame,
    uint32_t fw, uint32_t fh, uint32_t stride,
    int w_scaled, int h_scaled,
    int cells_x, int cells_y,
    double rotation_angle,
    uint8_t pixel_format,
    uint8_t detail, uint8_t threshold_steps,
    uint8_t* modes,
    uint8_t* patterns,
    uint8_t* fg_colors,
    uint8_t* bg_colors) {
    (void)frame; (void)fw; (void)fh; (void)stride;
    (void)w_scaled; (void)h_scaled; (void)rotation_angle; (void)pixel_format;
    (void)detail; (void)threshold_steps;
    memset(modes, 0, cells_x * cells_y);
    memset(patterns, 0, cells_x * cells_y);
    memset(fg_colors, 255, cells_x * cells_y * 3);
    memset(bg_colors, 0, cells_x * cells_y * 3);
}

extern "C" void render_blocks_cuda(
    const uint8_t* frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int w_scaled,
    int h_scaled,
    int cells_x,
    int cells_y,
    double rotation_angle,
    uint8_t pixel_format,
    uint8_t* fg_colors,
    uint8_t* bg_colors) {
    (void)frame_data; (void)frame_width; (void)frame_height; (void)frame_stride;
    (void)w_scaled; (void)h_scaled; (void)rotation_angle; (void)pixel_format;
    memset(fg_colors, 255, cells_x * cells_y * 3);
    memset(bg_colors, 0, cells_x * cells_y * 3);
}

extern "C" void render_ascii_cuda(
    const uint8_t* frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int w_scaled,
    int h_scaled,
    int cells_x,
    int cells_y,
    double rotation_angle,
    uint8_t pixel_format,
    uint8_t* intensities,
    uint8_t* fg_colors,
    uint8_t* bg_colors) {
    (void)frame_data; (void)frame_width; (void)frame_height; (void)frame_stride;
    (void)w_scaled; (void)h_scaled; (void)rotation_angle; (void)pixel_format;
    memset(intensities, 128, cells_x * cells_y);
    memset(fg_colors, 255, cells_x * cells_y * 3);
    memset(bg_colors, 0, cells_x * cells_y * 3);
}
