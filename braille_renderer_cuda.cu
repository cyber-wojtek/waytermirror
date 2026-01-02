#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// pixel color channel layout from compositor/capture source
enum PixelFormat : uint8_t {
  FMT_BGRx = 0,   // BGRX (wayland default, X ignored)
  FMT_BGRA = 1,   // BGRA with alpha
  FMT_RGBx = 2,   // RGBX (X ignored)
  FMT_RGBA = 3,   // RGBA with alpha
  FMT_xBGR = 4,   // XBGR (X padding first)
  FMT_ABGR = 5,   // ABGR (alpha first)
  FMT_xRGB = 6,   // XRGB (X padding first)
  FMT_ARGB = 7,   // ARGB (alpha first)
  FMT_BGR  = 8,   // BGR 24-bit (3 bytes)
  FMT_RGB  = 9,   // RGB 24-bit (3 bytes)
};

struct BrailleCellGPU {
  double lumas[8];
  uint8_t colors[8][3];
  double weights[8];
  bool has_edge;
  double mean_luma;
  double edge_strength;
};

// extract rgb from pixel data based on format
__device__ inline void get_rgb_cuda(const uint8_t* p, uint8_t& r, uint8_t& g, uint8_t& b, PixelFormat fmt) {
  switch (fmt) {
    case FMT_BGRx:
    case FMT_BGRA:
      b = p[0]; g = p[1]; r = p[2];
      break;
    case FMT_RGBx:
    case FMT_RGBA:
      r = p[0]; g = p[1]; b = p[2];
      break;
    case FMT_xBGR:
    case FMT_ABGR:
      b = p[1]; g = p[2]; r = p[3];
      break;
    case FMT_xRGB:
    case FMT_ARGB:
      r = p[1]; g = p[2]; b = p[3];
      break;
    case FMT_BGR:
      b = p[0]; g = p[1]; r = p[2];
      break;
    case FMT_RGB:
      r = p[0]; g = p[1]; b = p[2];
      break;
    default:
      b = p[0]; g = p[1]; r = p[2]; // fallback to BGRx
  }
}

__device__ inline double luma_cuda(uint8_t r, uint8_t g, uint8_t b) {
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

// bytes per pixel for format (3 for 24-bit, 4 for 32-bit)
__device__ inline int bpp_for_fmt(PixelFormat fmt) {
  return (fmt == FMT_BGR || fmt == FMT_RGB) ? 3 : 4;
}

__device__ inline void sample_rotated_pixel_cuda(
    const uint8_t* frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int x, int y,
    int rotated_width, int rotated_height,
    double rotation_angle,
    PixelFormat fmt,
    uint8_t& r, uint8_t& g, uint8_t& b)
{
    double rad = rotation_angle * M_PI / 180.0;
    double cos_a = cos(rad);
    double sin_a = sin(rad);
    
    double cx_out = rotated_width / 2.0;
    double cy_out = rotated_height / 2.0;
    double cx_in = frame_width / 2.0;
    double cy_in = frame_height / 2.0;
    
    double dx = x - cx_out;
    double dy = y - cy_out;
    
    double src_x_f = cos_a * dx + sin_a * dy + cx_in;
    double src_y_f = -sin_a * dx + cos_a * dy + cy_in;
    
    int src_x = (int)round(src_x_f);
    int src_y = (int)round(src_y_f);
    
    if (src_x < 0 || src_x >= (int)frame_width || src_y < 0 || src_y >= (int)frame_height) {
        r = g = b = 0;
        return;
    }
    
    const uint8_t* p = frame_data + src_y * frame_stride + src_x * bpp_for_fmt(fmt);

    get_rgb_cuda(p, r, g, b, fmt);
}

__device__ BrailleCellGPU analyze_braille_cell_cuda(
    const uint8_t* frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int cell_x,
    int cell_y,
    int w_scaled,
    int h_scaled,
    int rot_width,
    int rot_height,
    double rotation_angle,
    PixelFormat fmt,
    uint8_t detail_level) {
  
  BrailleCellGPU cell;
  memset(&cell, 0, sizeof(cell));
  
  // kernel size based on detail level
  int kernel_size;
  if (detail_level >= 95) {
    kernel_size = 1;
  } else if (detail_level >= 80) {
    kernel_size = 2;
  } else if (detail_level >= 60) {
    kernel_size = 3;
  } else if (detail_level >= 40) {
    kernel_size = 4;
  } else {
    kernel_size = 5;
  }
  
  // braille dot positions (2x4 grid)
  int dot_positions[8][2] = {
    {0, 0}, {0, 1}, {0, 2}, {1, 0},
    {1, 1}, {1, 2}, {0, 3}, {1, 3}
  };
  
  // sample each dot with gaussian kernel
  for (int dot = 0; dot < 8; dot++) {
    int dot_x = dot_positions[dot][0];
    int dot_y = dot_positions[dot][1];
    
    int rot_x = (cell_x * 2 + dot_x) * rot_width / w_scaled;
    int rot_y = (cell_y * 4 + dot_y) * rot_height / h_scaled;
    
    uint32_t r_sum = 0, g_sum = 0, b_sum = 0;
    double weight_sum = 0;
    
    for (int ky = 0; ky < kernel_size; ky++) {
      for (int kx = 0; kx < kernel_size; kx++) {
        int px = min(max(rot_x + kx - kernel_size/2, 0), rot_width - 1);
        int py = min(max(rot_y + ky - kernel_size/2, 0), rot_height - 1);
        
        uint8_t r, g, b;
        sample_rotated_pixel_cuda(frame_data, frame_width, frame_height, frame_stride,
                                   px, py, rot_width, rot_height, rotation_angle, fmt, r, g, b);
        
        double dx = kx - kernel_size / 2.0;
        double dy = ky - kernel_size / 2.0;
        double dist_sq = dx * dx + dy * dy;
        double sigma = kernel_size * 0.4;
        double weight = exp(-dist_sq / (2.0 * sigma * sigma));
        
        r_sum += r * weight;
        g_sum += g * weight;
        b_sum += b * weight;
        weight_sum += weight;
      }
    }
    
    uint8_t r = r_sum / weight_sum;
    uint8_t g = g_sum / weight_sum;
    uint8_t b = b_sum / weight_sum;
    
    cell.colors[dot][0] = r;
    cell.colors[dot][1] = g;
    cell.colors[dot][2] = b;
    cell.lumas[dot] = luma_cuda(r, g, b);
  }
  
  // edge detection via adjacent dot luma/color difference
  int adjacency_pairs[][2] = {
    {0, 1}, {1, 2}, {3, 4}, {4, 5}, {6, 7}, // vertical
    {0, 3}, {1, 4}, {2, 5}, {6, 7},         // horizontal
    {0, 4}, {1, 3}, {1, 5}, {2, 4}          // diagonal
  };
  
  cell.has_edge = false;
  double max_edge = 0;
  double edge_threshold = (detail_level >= 70) ? 30.0 : 50.0;
  
  for (int i = 0; i < 13; i++) {
    int idx1 = adjacency_pairs[i][0];
    int idx2 = adjacency_pairs[i][1];
    
    double dr = cell.colors[idx1][0] - cell.colors[idx2][0];
    double dg = cell.colors[idx1][1] - cell.colors[idx2][1];
    double db = cell.colors[idx1][2] - cell.colors[idx2][2];
    double color_dist = sqrt(2.0 * dr * dr + 4.0 * dg * dg + 3.0 * db * db);
    
    double luma_diff = fabs(cell.lumas[idx1] - cell.lumas[idx2]);
    double edge = fmax(color_dist, luma_diff * 2.0);
    
    max_edge = fmax(max_edge, edge);
    if (edge > edge_threshold) {
      cell.has_edge = true;
    }
  }
  
  cell.edge_strength = max_edge;
  
  // mean luma
  double sum = 0;
  for (int i = 0; i < 8; i++) {
    sum += cell.lumas[i];
  }
  cell.mean_luma = sum / 8.0;
  
  // contrast-based weights
  for (int i = 0; i < 8; i++) {
    double contrast = fabs(cell.lumas[i] - cell.mean_luma);
    cell.weights[i] = 1.0 + (contrast / 128.0);
  }
  
  return cell;
}

__global__ void render_blocks_kernel(
    const uint8_t* frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int w_scaled,
    int h_scaled,
    int cells_x,
    int cells_y,
    int rot_width,
    int rot_height,
    double rotation_angle,
    PixelFormat fmt,
    uint8_t* fg_colors,
    uint8_t* bg_colors)
{
    int cell_x = blockIdx.x * blockDim.x + threadIdx.x;
    int cell_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (cell_x >= cells_x || cell_y >= cells_y) return;
    
    int idx = cell_y * cells_x + cell_x;
    
    // Map block position to rotated image coordinates
    // Top half: sample at cell_y/cells_y of image height
    // Bottom half: sample at (cell_y+0.5)/cells_y of image height
    int rot_x = cell_x * rot_width / cells_x;
    int rot_y_top = (cell_y * 2) * rot_height / (cells_y * 2);
    int rot_y_bot = (cell_y * 2 + 1) * rot_height / (cells_y * 2);
    
    rot_x = min(max(rot_x, 0), rot_width - 1);
    rot_y_top = min(max(rot_y_top, 0), rot_height - 1);
    rot_y_bot = min(max(rot_y_bot, 0), rot_height - 1);
    
    uint8_t r_top, g_top, b_top;
    sample_rotated_pixel_cuda(frame_data, frame_width, frame_height, frame_stride,
                               rot_x, rot_y_top, rot_width, rot_height, rotation_angle, fmt, r_top, g_top, b_top);
    
    uint8_t r_bot, g_bot, b_bot;
    sample_rotated_pixel_cuda(frame_data, frame_width, frame_height, frame_stride,
                               rot_x, rot_y_bot, rot_width, rot_height, rotation_angle, fmt, r_bot, g_bot, b_bot);
    
    fg_colors[idx * 3 + 0] = r_top;
    fg_colors[idx * 3 + 1] = g_top;
    fg_colors[idx * 3 + 2] = b_top;
    bg_colors[idx * 3 + 0] = r_bot;
    bg_colors[idx * 3 + 1] = g_bot;
    bg_colors[idx * 3 + 2] = b_bot;
}

__global__ void render_ascii_kernel(
    const uint8_t* frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int w_scaled,
    int h_scaled,
    int cells_x,
    int cells_y,
    int rot_width,
    int rot_height,
    double rotation_angle,
    PixelFormat fmt,
    uint8_t* intensities,
    uint8_t* fg_colors,
    uint8_t* bg_colors)
{
    int cell_x = blockIdx.x * blockDim.x + threadIdx.x;
    int cell_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (cell_x >= cells_x || cell_y >= cells_y) return;
    
    int idx = cell_y * cells_x + cell_x;
    
    // sample cell center
    int rot_x = (cell_x * 2 + 1) * rot_width / w_scaled;
    int rot_y = (cell_y * 4 + 2) * rot_height / h_scaled;
    
    rot_x = min(max(rot_x, 0), rot_width - 1);
    rot_y = min(max(rot_y, 0), rot_height - 1);
    
    uint8_t r, g, b;
    sample_rotated_pixel_cuda(frame_data, frame_width, frame_height, frame_stride,
                               rot_x, rot_y, rot_width, rot_height, rotation_angle, fmt, r, g, b);
    
    double luma = luma_cuda(r, g, b);
    intensities[idx] = (uint8_t)luma;
    
    fg_colors[idx * 3 + 0] = r;
    fg_colors[idx * 3 + 1] = g;
    fg_colors[idx * 3 + 2] = b;
    bg_colors[idx * 3 + 0] = 0;
    bg_colors[idx * 3 + 1] = 0;
    bg_colors[idx * 3 + 2] = 0;
}

struct RegionalColorAnalysisGPU {
  uint8_t dominant_fg_r, dominant_fg_g, dominant_fg_b;
  uint8_t dominant_bg_r, dominant_bg_g, dominant_bg_b;
  double contrast;
};

__device__ RegionalColorAnalysisGPU analyze_regional_colors_cuda(
    const uint8_t* frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int cell_x,
    int cell_y,
    int w_scaled,
    int h_scaled,
    int cells_x,
    int cells_y,
    int rot_width,
    int rot_height,
    double rotation_angle,
    PixelFormat fmt,
    uint8_t detail_level) {
  
  int region_size = 10;
  
  uint8_t reds[1000], greens[1000], blues[1000], lumas_arr[1000];
  int sample_count = 0;
  int half_region = region_size / 2;
  
  for (int dy = -half_region; dy <= half_region && sample_count < 1000; dy++) {
    for (int dx = -half_region; dx <= half_region && sample_count < 1000; dx++) {
      int nx = cell_x + dx;
      int ny = cell_y + dy;
      
      if (nx < 0 || nx >= cells_x || ny < 0 || ny >= cells_y) continue;
      
      // sample all 8 braille dots in cell
      for (int dot = 0; dot < 8; dot++) {
        int dot_positions[8][2] = {
          {0, 0}, {0, 1}, {0, 2}, {1, 0},
          {1, 1}, {1, 2}, {0, 3}, {1, 3}
        };
        
        int dot_x = dot_positions[dot][0];
        int dot_y = dot_positions[dot][1];
        
        int rot_x = (nx * 2 + dot_x) * rot_width / w_scaled;
        int rot_y = (ny * 4 + dot_y) * rot_height / h_scaled;
        
        rot_x = min(max(rot_x, 0), rot_width - 1);
        rot_y = min(max(rot_y, 0), rot_height - 1);
        
        uint8_t r, g, b;
        sample_rotated_pixel_cuda(frame_data, frame_width, frame_height, frame_stride,
                                   rot_x, rot_y, rot_width, rot_height, rotation_angle, fmt, r, g, b);
        
        reds[sample_count] = r;
        greens[sample_count] = g;
        blues[sample_count] = b;
        lumas_arr[sample_count] = luma_cuda(r, g, b);
        sample_count++;
      }
    }
  }
  
  if (sample_count == 0) {
    return {128, 128, 128, 64, 64, 64, 64.0};
  }
  
  // find median luma via bubble sort (small arrays)
  uint8_t sorted_lumas[1000];
  memcpy(sorted_lumas, lumas_arr, sample_count);
  for (int i = 0; i < sample_count - 1; i++) {
    for (int j = 0; j < sample_count - i - 1; j++) {
      if (sorted_lumas[j] > sorted_lumas[j + 1]) {
        uint8_t tmp = sorted_lumas[j];
        sorted_lumas[j] = sorted_lumas[j + 1];
        sorted_lumas[j + 1] = tmp;
      }
    }
  }
  uint8_t median_luma = sorted_lumas[sample_count / 2];
  
  // split into fg (bright) / bg (dark) groups
  uint32_t fg_r_sum = 0, fg_g_sum = 0, fg_b_sum = 0, fg_count = 0;
  uint32_t bg_r_sum = 0, bg_g_sum = 0, bg_b_sum = 0, bg_count = 0;
  
  for (int i = 0; i < sample_count; i++) {
    if (lumas_arr[i] >= median_luma) {
      fg_r_sum += reds[i];
      fg_g_sum += greens[i];
      fg_b_sum += blues[i];
      fg_count++;
    } else {
      bg_r_sum += reds[i];
      bg_g_sum += greens[i];
      bg_b_sum += blues[i];
      bg_count++;
    }
  }
  
  RegionalColorAnalysisGPU result;
  
  if (fg_count > 0) {
    result.dominant_fg_r = fg_r_sum / fg_count;
    result.dominant_fg_g = fg_g_sum / fg_count;
    result.dominant_fg_b = fg_b_sum / fg_count;
  } else {
    result.dominant_fg_r = result.dominant_fg_g = result.dominant_fg_b = 200;
  }
  
  if (bg_count > 0) {
    result.dominant_bg_r = bg_r_sum / bg_count;
    result.dominant_bg_g = bg_g_sum / bg_count;
    result.dominant_bg_b = bg_b_sum / bg_count;
  } else {
    result.dominant_bg_r = result.dominant_bg_g = result.dominant_bg_b = 55;
  }
  
  double fg_luma = luma_cuda(result.dominant_fg_r, result.dominant_fg_g, result.dominant_fg_b);
  double bg_luma = luma_cuda(result.dominant_bg_r, result.dominant_bg_g, result.dominant_bg_b);
  result.contrast = fabs(fg_luma - bg_luma);
  
  return result;
}

// threshold the 8 braille dots into a pattern byte
__device__ uint8_t calculate_braille_pattern_cuda(
    const BrailleCellGPU& cell,
    uint8_t detail_level,
    uint8_t quality,
    const RegionalColorAnalysisGPU& regional) {
  
  int step = max(1, 16 - (quality / 7));
  double threshold;
  
  // otsu-ish thresholding for high detail, mean luma otherwise
  if (detail_level >= 70) {
    double best_threshold = cell.mean_luma;
    double best_separation = 0;
    
    for (int t_int = 0; t_int <= 255; t_int += step) {
      double t = t_int;
      double sum_below = 0, sum_above = 0;
      int count_below = 0, count_above = 0;
      
      for (int i = 0; i < 8; i++) {
        if (cell.lumas[i] < t) {
          sum_below += cell.lumas[i];
          count_below++;
        } else {
          sum_above += cell.lumas[i];
          count_above++;
        }
      }
      
      if (count_below > 0 && count_above > 0) {
        double mean_below = sum_below / count_below;
        double mean_above = sum_above / count_above;
        double local_contrast = mean_above - mean_below;
        double separation = count_below * count_above * local_contrast * local_contrast;
        
        if (separation > best_separation) {
          best_separation = separation;
          best_threshold = t;
        }
      }
    }
    threshold = best_threshold;
  } else {
    threshold = cell.mean_luma;
  }
  
  threshold = fmin(fmax(threshold, 0.0), 255.0);
  
  uint8_t pattern = 0;
  for (int dot = 0; dot < 8; dot++) {
    if (cell.lumas[dot] > threshold) {
      pattern |= (1 << dot);
    }
  }
  
  // invert if most dots are lit (dark text on light bg -> foreground)
  int lit_count = __popc(pattern);
  if (lit_count > 4) {
    pattern = ~pattern;
  }
  
  return pattern;
}

// average lit/unlit dot colors to get fg/bg
__device__ void calculate_braille_colors_cuda(
    const BrailleCellGPU& cell,
    uint8_t pattern,
    uint8_t& fg_r, uint8_t& fg_g, uint8_t& fg_b,
    uint8_t& bg_r, uint8_t& bg_g, uint8_t& bg_b,
    uint8_t detail_level,
    const RegionalColorAnalysisGPU& regional) {
  
  int lit_count = 0, unlit_count = 0;
  uint32_t lit_r = 0, lit_g = 0, lit_b = 0;
  uint32_t unlit_r = 0, unlit_g = 0, unlit_b = 0;
  
  for (int i = 0; i < 8; i++) {
    if (pattern & (1 << i)) {
      lit_r += cell.colors[i][0];
      lit_g += cell.colors[i][1];
      lit_b += cell.colors[i][2];
      lit_count++;
    } else {
      unlit_r += cell.colors[i][0];
      unlit_g += cell.colors[i][1];
      unlit_b += cell.colors[i][2];
      unlit_count++;
    }
  }
  
  // no lit dots? use regional bg color
  if (lit_count == 0) {
    fg_r = regional.dominant_bg_r;
    fg_g = regional.dominant_bg_g;
    fg_b = regional.dominant_bg_b;
    bg_r = fg_r;
    bg_g = fg_g;
    bg_b = fg_b;
    return;
  }
  // all lit? use regional fg color
  if (unlit_count == 0) {
    fg_r = regional.dominant_fg_r;
    fg_g = regional.dominant_fg_g;
    fg_b = regional.dominant_fg_b;
    bg_r = fg_r;
    bg_g = fg_g;
    bg_b = fg_b;
    return;
  }
  
  fg_r = lit_r / lit_count;
  fg_g = lit_g / lit_count;
  fg_b = lit_b / lit_count;
  
  bg_r = unlit_r / unlit_count;
  bg_g = unlit_g / unlit_count;
  bg_b = unlit_b / unlit_count;
}

__global__ void render_braille_kernel(
    const uint8_t* frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int w_scaled,
    int h_scaled,
    int cells_x,
    int cells_y,
    int rot_width,
    int rot_height,
    double rotation_angle,
    PixelFormat fmt,
    uint8_t detail_level,
    uint8_t threshold_steps,
    uint8_t* patterns,
    uint8_t* fg_colors,
    uint8_t* bg_colors)
{
    int cell_x = blockIdx.x * blockDim.x + threadIdx.x;
    int cell_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (cell_x >= cells_x || cell_y >= cells_y) return;
    
    int idx = cell_y * cells_x + cell_x;
    
    BrailleCellGPU cell = analyze_braille_cell_cuda(
        frame_data, frame_width, frame_height, frame_stride,
        cell_x, cell_y, w_scaled, h_scaled, rot_width, rot_height, rotation_angle, fmt, detail_level);
    
    RegionalColorAnalysisGPU regional = analyze_regional_colors_cuda(
        frame_data, frame_width, frame_height, frame_stride,
        cell_x, cell_y, w_scaled, h_scaled, cells_x, cells_y, rot_width, rot_height, rotation_angle, fmt, detail_level);
    
    uint8_t pattern = calculate_braille_pattern_cuda(cell, detail_level, threshold_steps, regional);
    
    uint8_t fg_r, fg_g, fg_b, bg_r, bg_g, bg_b;
    calculate_braille_colors_cuda(cell, pattern, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b,
                                  detail_level, regional);
    
    patterns[idx] = pattern;
    fg_colors[idx * 3 + 0] = fg_r;
    fg_colors[idx * 3 + 1] = fg_g;
    fg_colors[idx * 3 + 2] = fg_b;
    bg_colors[idx * 3 + 0] = bg_r;
    bg_colors[idx * 3 + 1] = bg_g;
    bg_colors[idx * 3 + 2] = bg_b;
}

// hybrid: braille where edges detected, half-blocks elsewhere
__global__ void render_hybrid_kernel(
    const uint8_t* frame,
    uint32_t fw, uint32_t fh, uint32_t stride,
    int w_scaled, int h_scaled,
    int cells_x, int cells_y,
    int rot_width, int rot_height,
    double rotation_angle,
    PixelFormat fmt,
    uint8_t detail, uint8_t threshold_steps,
    uint8_t* modes,
    uint8_t* patterns,
    uint8_t* fg_colors,
    uint8_t* bg_colors)
{
    int cell_x = blockIdx.x * blockDim.x + threadIdx.x;
    int cell_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (cell_x >= cells_x || cell_y >= cells_y) return;
    
    int idx = cell_y * cells_x + cell_x;
    
    BrailleCellGPU cell = analyze_braille_cell_cuda(
        frame, fw, fh, stride, cell_x, cell_y, w_scaled, h_scaled, rot_width, rot_height, rotation_angle, fmt, detail);
    
    RegionalColorAnalysisGPU regional = analyze_regional_colors_cuda(
        frame, fw, fh, stride, cell_x, cell_y, w_scaled, h_scaled,
        cells_x, cells_y, rot_width, rot_height, rotation_angle, fmt, detail);
    
    // braille if edge detected + sufficient detail level
    bool use_braille = cell.has_edge && detail >= 60;
    modes[idx] = use_braille ? 1 : 0;
    
    if (use_braille) {
        uint8_t pattern = calculate_braille_pattern_cuda(cell, detail, threshold_steps, regional);
        uint8_t fg_r, fg_g, fg_b, bg_r, bg_g, bg_b;
        calculate_braille_colors_cuda(cell, pattern, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b,
                                      detail, regional);
        patterns[idx] = pattern;
        fg_colors[idx * 3 + 0] = fg_r;
        fg_colors[idx * 3 + 1] = fg_g;
        fg_colors[idx * 3 + 2] = fg_b;
        bg_colors[idx * 3 + 0] = bg_r;
        bg_colors[idx * 3 + 1] = bg_g;
        bg_colors[idx * 3 + 2] = bg_b;
    } else {
        // blocks: top 6 dots -> fg, bottom 2 -> bg
        uint32_t r_top = 0, g_top = 0, b_top = 0;
        uint32_t r_bot = 0, g_bot = 0, b_bot = 0;
        
        for (int i = 0; i < 6; i++) {
            r_top += cell.colors[i][0];
            g_top += cell.colors[i][1];
            b_top += cell.colors[i][2];
        }
        for (int i = 6; i < 8; i++) {
            r_bot += cell.colors[i][0];
            g_bot += cell.colors[i][1];
            b_bot += cell.colors[i][2];
        }
        
        fg_colors[idx * 3 + 0] = r_top / 6;
        fg_colors[idx * 3 + 1] = g_top / 6;
        fg_colors[idx * 3 + 2] = b_top / 6;
        bg_colors[idx * 3 + 0] = r_bot / 2;
        bg_colors[idx * 3 + 1] = g_bot / 2;
        bg_colors[idx * 3 + 2] = b_bot / 2;
        patterns[idx] = 0;
    }
}


extern "C" void render_braille_cuda(
    const uint8_t *frame_data,
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
    uint8_t *patterns,
    uint8_t *fg_colors,
    uint8_t *bg_colors)
{
    PixelFormat fmt = (PixelFormat)pixel_format;
    uint32_t rot_width = frame_width;
    uint32_t rot_height = frame_height;
    if (rotation_angle != 0) {
        double rad = fabs(rotation_angle) * M_PI / 180.0;
        double cos_a = fabs(cos(rad));
        double sin_a = fabs(sin(rad));
        rot_width = (uint32_t)ceil(frame_width * cos_a + frame_height * sin_a);
        rot_height = (uint32_t)ceil(frame_width * sin_a + frame_height * cos_a);
    }
    
    dim3 block(16, 16);
    dim3 grid((cells_x + 15) / 16, (cells_y + 15) / 16);
    
    render_braille_kernel<<<grid, block>>>(
        frame_data, frame_width, frame_height, frame_stride,
        w_scaled, h_scaled, cells_x, cells_y, rot_width, rot_height, rotation_angle, fmt,
        detail_level, threshold_steps, patterns, fg_colors, bg_colors);
    
    cudaDeviceSynchronize();
}

extern "C" void render_hybrid_cuda(
    const uint8_t *frame,
    uint32_t fw, uint32_t fh, uint32_t stride,
    int w_scaled, int h_scaled,
    int cells_x, int cells_y,
    double rotation_angle,
    uint8_t pixel_format,
    uint8_t detail, uint8_t threshold_steps,
    uint8_t *modes,
    uint8_t *patterns,
    uint8_t *fg_colors,
    uint8_t *bg_colors)
{
    PixelFormat fmt = (PixelFormat)pixel_format;
    uint32_t rot_width = fw;
    uint32_t rot_height = fh;
    if (rotation_angle != 0) {
        double rad = fabs(rotation_angle) * M_PI / 180.0;
        double cos_a = fabs(cos(rad));
        double sin_a = fabs(sin(rad));
        rot_width = (uint32_t)ceil(fw * cos_a + fh * sin_a);
        rot_height = (uint32_t)ceil(fw * sin_a + fh * cos_a);
    }
    
    dim3 block(16, 16);
    dim3 grid((cells_x + 15) / 16, (cells_y + 15) / 16);

    render_hybrid_kernel<<<grid, block>>>(
        frame, fw, fh, stride, w_scaled, h_scaled, cells_x, cells_y,
        rot_width, rot_height, rotation_angle, fmt, detail, threshold_steps,
        modes, patterns, fg_colors, bg_colors);

    cudaDeviceSynchronize();
}

extern "C" void render_blocks_cuda(
    const uint8_t *frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int w_scaled,
    int h_scaled,
    int cells_x,
    int cells_y,
    double rotation_angle,
    uint8_t pixel_format,
    uint8_t *fg_colors,
    uint8_t *bg_colors)
{
    PixelFormat fmt = (PixelFormat)pixel_format;
    uint32_t rot_width = frame_width;
    uint32_t rot_height = frame_height;
    if (rotation_angle != 0) {
        double rad = fabs(rotation_angle) * M_PI / 180.0;
        double cos_a = fabs(cos(rad));
        double sin_a = fabs(sin(rad));
        rot_width = (uint32_t)ceil(frame_width * cos_a + frame_height * sin_a);
        rot_height = (uint32_t)ceil(frame_width * sin_a + frame_height * cos_a);
    }
    
    dim3 block(16, 16);
    dim3 grid((cells_x + 15) / 16, (cells_y + 15) / 16);
    
    render_blocks_kernel<<<grid, block>>>(
        frame_data, frame_width, frame_height, frame_stride,
        w_scaled, h_scaled, cells_x, cells_y, rot_width, rot_height, rotation_angle, fmt,
        fg_colors, bg_colors);
    
    cudaDeviceSynchronize();
}

extern "C" void render_ascii_cuda(
    const uint8_t *frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int w_scaled,
    int h_scaled,
    int cells_x,
    int cells_y,
    double rotation_angle,
    uint8_t pixel_format,
    uint8_t *intensities,
    uint8_t *fg_colors,
    uint8_t *bg_colors)
{
    PixelFormat fmt = (PixelFormat)pixel_format;
    uint32_t rot_width = frame_width;
    uint32_t rot_height = frame_height;
    if (rotation_angle != 0) {
        double rad = fabs(rotation_angle) * M_PI / 180.0;
        double cos_a = fabs(cos(rad));
        double sin_a = fabs(sin(rad));
        rot_width = (uint32_t)ceil(frame_width * cos_a + frame_height * sin_a);
        rot_height = (uint32_t)ceil(frame_width * sin_a + frame_height * cos_a);
    }
    
    dim3 block(16, 16);
    dim3 grid((cells_x + 15) / 16, (cells_y + 15) / 16);
    
    render_ascii_kernel<<<grid, block>>>(
        frame_data, frame_width, frame_height, frame_stride,
        w_scaled, h_scaled, cells_x, cells_y, rot_width, rot_height, rotation_angle, fmt,
        intensities, fg_colors, bg_colors);
    
    cudaDeviceSynchronize();
}
