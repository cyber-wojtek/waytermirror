#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// CUDA KERNEL HELPERS
// ============================================================================

__device__ inline void get_rgb_cuda(const uint8_t* p, uint8_t& r, uint8_t& g, uint8_t& b) {
  b = p[0];
  g = p[1];
  r = p[2];
}

__device__ inline double luma_cuda(uint8_t r, uint8_t g, uint8_t b) {
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

// Helper to sample a pixel with rotation transformation
__device__ inline void sample_rotated_pixel_cuda(
    const uint8_t* frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int x, int y,
    int rotated_width, int rotated_height,
    double rotation_angle,
    uint8_t& r, uint8_t& g, uint8_t& b)
{
    // Convert angle to radians
    double rad = rotation_angle * M_PI / 180.0;
    double cos_a = cos(rad);
    double sin_a = sin(rad);
    
    // Center of output (rotated) space
    double cx_out = rotated_width / 2.0;
    double cy_out = rotated_height / 2.0;
    
    // Center of input (original) space
    double cx_in = frame_width / 2.0;
    double cy_in = frame_height / 2.0;
    
    // Translate to origin, rotate backwards, translate back
    double dx = x - cx_out;
    double dy = y - cy_out;
    
    // Inverse rotation (rotate point back to find source)
    double src_x_f = cos_a * dx + sin_a * dy + cx_in;
    double src_y_f = -sin_a * dx + cos_a * dy + cy_in;
    
    int src_x = (int)round(src_x_f);
    int src_y = (int)round(src_y_f);
    
    // Check bounds - return black for out-of-bounds
    if (src_x < 0 || src_x >= (int)frame_width || src_y < 0 || src_y >= (int)frame_height) {
        r = g = b = 0;
        return;
    }
    
    const uint8_t* p = frame_data + src_y * frame_stride + src_x * 4;
    get_rgb_cuda(p, r, g, b);
}

// ============================================================================
// BRAILLE CELL ANALYSIS (GPU)
// ============================================================================

struct BrailleCellGPU {
  double lumas[8];
  uint8_t colors[8][3];
  double mean_luma;
  bool has_edge;
  double edge_strength;
};

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
    uint8_t detail_level) {
  
  BrailleCellGPU cell;
  memset(&cell, 0, sizeof(cell));
  
  // Braille dot positions (2x4 grid)
  int dot_positions[8][2] = {
    {0, 0}, {0, 1}, {0, 2}, {1, 0},
    {1, 1}, {1, 2}, {0, 3}, {1, 3}
  };
  
  // Calculate sampling kernel size
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
    kernel_size = 2;
  }
  
  // Sample each dot with Gaussian kernel
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
                                  px, py, rot_width, rot_height, rotation_angle, r, g, b);
        
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
  
  // Edge detection
  int adjacency_pairs[][2] = {
    {0, 1}, {1, 2}, {3, 4}, {4, 5}, {6, 7},
    {0, 3}, {1, 4}, {2, 5}, {6, 7},
    {0, 4}, {1, 3}, {1, 5}, {2, 4}
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
  
  // Calculate mean luma
  double sum = 0;
  for (int i = 0; i < 8; i++) {
    sum += cell.lumas[i];
  }
  cell.mean_luma = sum / 8.0;
  
  return cell;
}

// ============================================================================
// REGIONAL COLOR ANALYSIS (GPU)
// ============================================================================

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
    int rot_width,
    int rot_height,
    double rotation_angle,
    int cells_x,
    int cells_y,
    uint8_t detail_level) {
  
  int region_size;
  if (detail_level >= 90) {
    region_size = 3;
  } else if (detail_level < 50) {
    region_size = 7;
  } else {
    region_size = 5;
  }
  
  // Collect colors from region
  uint8_t reds[300], greens[300], blues[300], lumas[300];
  int sample_count = 0;
  int half_region = region_size / 2;
  
  for (int dy = -half_region; dy <= half_region && sample_count < 300; dy++) {
    for (int dx = -half_region; dx <= half_region && sample_count < 300; dx++) {
      int nx = cell_x + dx;
      int ny = cell_y + dy;
      
      if (nx < 0 || nx >= cells_x || ny < 0 || ny >= cells_y) continue;
      
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
                                  rot_x, rot_y, rot_width, rot_height, rotation_angle, r, g, b);
        
        reds[sample_count] = r;
        greens[sample_count] = g;
        blues[sample_count] = b;
        lumas[sample_count] = luma_cuda(r, g, b);
        sample_count++;
      }
    }
  }
  
  if (sample_count == 0) {
    return {128, 128, 128, 64, 64, 64, 64.0};
  }
  
  // Find median luminance (simple bubble sort for small arrays)
  uint8_t sorted_lumas[300];
  memcpy(sorted_lumas, lumas, sample_count);
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
  
  // Split into light/dark groups
  uint32_t fg_r_sum = 0, fg_g_sum = 0, fg_b_sum = 0, fg_count = 0;
  uint32_t bg_r_sum = 0, bg_g_sum = 0, bg_b_sum = 0, bg_count = 0;
  
  for (int i = 0; i < sample_count; i++) {
    if (lumas[i] >= median_luma) {
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

// ============================================================================
// BRAILLE PATTERN CALCULATION (GPU)
// ============================================================================

__device__ uint8_t calculate_braille_pattern_cuda(
    const BrailleCellGPU& cell,
    uint8_t detail_level,
    uint8_t quality,
    const RegionalColorAnalysisGPU& regional) {
  
  int step = max(1, 16 - (quality / 7));
  
  // High detail: Otsu thresholding
  if (detail_level >= 90) {
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
        double separation = count_below * count_above * 
                           (mean_above - mean_below) * (mean_above - mean_below);
        
        if (separation > best_separation) {
          best_separation = separation;
          best_threshold = t;
        }
      }
    }
    
    uint8_t pattern = 0;
    for (int dot = 0; dot < 8; dot++) {
      if (cell.lumas[dot] > best_threshold) {
        pattern |= (1 << dot);
      }
    }
    
    // Invert if majority lit
    int lit_count = __popc(pattern);
    if (lit_count > 4) {
      pattern = ~pattern;
    }
    
    return pattern;
  }
  
  // Medium-high detail
  double threshold;
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
  
  // Invert if majority lit
  int lit_count = __popc(pattern);
  if (lit_count > 4) {
    pattern = ~pattern;
  }
  
  return pattern;
}

// ============================================================================
// COLOR CALCULATION (GPU)
// ============================================================================

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
  
  if (lit_count == 0) {
    fg_r = regional.dominant_bg_r;
    fg_g = regional.dominant_bg_g;
    fg_b = regional.dominant_bg_b;
    bg_r = fg_r; bg_g = fg_g; bg_b = fg_b;
    return;
  } else if (unlit_count == 0) {
    fg_r = regional.dominant_fg_r;
    fg_g = regional.dominant_fg_g;
    fg_b = regional.dominant_fg_b;
    bg_r = fg_r; bg_g = fg_g; bg_b = fg_b;
    return;
  }
  
  fg_r = lit_r / lit_count;
  fg_g = lit_g / lit_count;
  fg_b = lit_b / lit_count;
  
  bg_r = unlit_r / unlit_count;
  bg_g = unlit_g / unlit_count;
  bg_b = unlit_b / unlit_count;
}

// ============================================================================
// CUDA KERNEL: BRAILLE RENDERING
// ============================================================================

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
    uint8_t detail_level,
    uint8_t threshold_steps,
    uint8_t* patterns,
    uint8_t* fg_colors,
    uint8_t* bg_colors) {
  
  int cx = blockIdx.x * blockDim.x + threadIdx.x;
  int cy = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (cx >= cells_x || cy >= cells_y) return;
  
  int idx = cy * cells_x + cx;
  
  // Analyze cell
  BrailleCellGPU cell = analyze_braille_cell_cuda(
    frame_data, frame_width, frame_height, frame_stride,
    cx, cy, w_scaled, h_scaled, rot_width, rot_height, rotation_angle, detail_level
  );
  
  // Regional analysis
  RegionalColorAnalysisGPU regional = analyze_regional_colors_cuda(
    frame_data, frame_width, frame_height, frame_stride,
    cx, cy, w_scaled, h_scaled, rot_width, rot_height, rotation_angle, cells_x, cells_y, detail_level
  );
  
  // Calculate pattern
  uint8_t pattern = calculate_braille_pattern_cuda(
    cell, detail_level, threshold_steps, regional
  );
  
  // Calculate colors
  uint8_t fg_r, fg_g, fg_b, bg_r, bg_g, bg_b;
  calculate_braille_colors_cuda(
    cell, pattern, fg_r, fg_g, fg_b, 
    bg_r, bg_g, bg_b, detail_level, regional
  );
  
  // Write output
  patterns[idx] = pattern;
  fg_colors[idx * 3 + 0] = fg_r;
  fg_colors[idx * 3 + 1] = fg_g;
  fg_colors[idx * 3 + 2] = fg_b;
  bg_colors[idx * 3 + 0] = bg_r;
  bg_colors[idx * 3 + 1] = bg_g;
  bg_colors[idx * 3 + 2] = bg_b;
}

// ============================================================================
// CUDA KERNEL: HYBRID RENDERING
// ============================================================================

__global__ void render_hybrid_kernel(
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
    uint8_t detail_level,
    uint8_t threshold_steps,
    uint8_t* modes,
    uint8_t* patterns,
    uint8_t* fg_colors,
    uint8_t* bg_colors) {
  
  int cx = blockIdx.x * blockDim.x + threadIdx.x;
  int cy = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (cx >= cells_x || cy >= cells_y) return;
  
  int idx = cy * cells_x + cx;
  
  // Analyze cell
  BrailleCellGPU cell = analyze_braille_cell_cuda(
    frame_data, frame_width, frame_height, frame_stride,
    cx, cy, w_scaled, h_scaled, rot_width, rot_height, rotation_angle, detail_level
  );
  
  // Decide mode based on edge detection
  bool use_braille = cell.has_edge;
  modes[idx] = use_braille ? 1 : 0;
  
  if (use_braille) {
    // High-detail braille rendering
    RegionalColorAnalysisGPU regional = analyze_regional_colors_cuda(
      frame_data, frame_width, frame_height, frame_stride,
      cx, cy, w_scaled, h_scaled, rot_width, rot_height, rotation_angle, cells_x, cells_y, detail_level
    );
    
    uint8_t pattern = calculate_braille_pattern_cuda(
      cell, detail_level, threshold_steps, regional
    );
    
    uint8_t fg_r, fg_g, fg_b, bg_r, bg_g, bg_b;
    calculate_braille_colors_cuda(
      cell, pattern, fg_r, fg_g, fg_b, 
      bg_r, bg_g, bg_b, detail_level, regional
    );
    
    patterns[idx] = pattern;
    fg_colors[idx * 3 + 0] = fg_r;
    fg_colors[idx * 3 + 1] = fg_g;
    fg_colors[idx * 3 + 2] = fg_b;
    bg_colors[idx * 3 + 0] = bg_r;
    bg_colors[idx * 3 + 1] = bg_g;
    bg_colors[idx * 3 + 2] = bg_b;
  } else {
    // Half-block rendering for flat areas
    uint32_t top_r = 0, top_g = 0, top_b = 0;
    uint32_t bot_r = 0, bot_g = 0, bot_b = 0;
    
    // Top half (dots 0,1,3,4)
    for (int i : {0, 1, 3, 4}) {
      top_r += cell.colors[i][0];
      top_g += cell.colors[i][1];
      top_b += cell.colors[i][2];
    }
    
    // Bottom half (dots 2,5,6,7)
    for (int i : {2, 5, 6, 7}) {
      bot_r += cell.colors[i][0];
      bot_g += cell.colors[i][1];
      bot_b += cell.colors[i][2];
    }
    
    patterns[idx] = 0;  // Not used for half-blocks
    fg_colors[idx * 3 + 0] = top_r / 4;
    fg_colors[idx * 3 + 1] = top_g / 4;
    fg_colors[idx * 3 + 2] = top_b / 4;
    bg_colors[idx * 3 + 0] = bot_r / 4;
    bg_colors[idx * 3 + 1] = bot_g / 4;
    bg_colors[idx * 3 + 2] = bot_b / 4;
  }
}

// ============================================================================
// HOST WRAPPER FUNCTIONS
// ============================================================================

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
    uint8_t detail_level,
    uint8_t threshold_steps,
    uint8_t* patterns,
    uint8_t* fg_colors,
    uint8_t* bg_colors) {
  
  // Calculate rotated dimensions
  double rad = rotation_angle * M_PI / 180.0;
  double cos_a = cos(fabs(rad));
  double sin_a = sin(fabs(rad));
  int rot_width = (int)ceil(frame_width * cos_a + frame_height * sin_a);
  int rot_height = (int)ceil(frame_width * sin_a + frame_height * cos_a);
  if (fmod(fabs(rotation_angle), 360.0) == 0.0) {
    rot_width = frame_width;
    rot_height = frame_height;
  }
  
  // Allocate device memory
  uint8_t *d_frame, *d_patterns, *d_fg_colors, *d_bg_colors;
  size_t frame_size = frame_height * frame_stride;
  size_t output_size = cells_x * cells_y;
  
  cudaMalloc(&d_frame, frame_size);
  cudaMalloc(&d_patterns, output_size);
  cudaMalloc(&d_fg_colors, output_size * 3);
  cudaMalloc(&d_bg_colors, output_size * 3);
  
  // Copy input to device
  cudaMemcpy(d_frame, frame_data, frame_size, cudaMemcpyHostToDevice);
  
  // Launch kernel
  dim3 block(16, 16);
  dim3 grid((cells_x + block.x - 1) / block.x, 
            (cells_y + block.y - 1) / block.y);
  
  render_braille_kernel<<<grid, block>>>(
    d_frame, frame_width, frame_height, frame_stride,
    w_scaled, h_scaled, cells_x, cells_y,
    rot_width, rot_height, rotation_angle,
    detail_level, threshold_steps,
    d_patterns, d_fg_colors, d_bg_colors
  );
  
  // Copy results back
  cudaMemcpy(patterns, d_patterns, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(fg_colors, d_fg_colors, output_size * 3, cudaMemcpyDeviceToHost);
  cudaMemcpy(bg_colors, d_bg_colors, output_size * 3, cudaMemcpyDeviceToHost);
  
  // Free device memory
  cudaFree(d_frame);
  cudaFree(d_patterns);
  cudaFree(d_fg_colors);
  cudaFree(d_bg_colors);
}

extern "C" void render_hybrid_cuda(
    const uint8_t* frame,
    uint32_t fw, uint32_t fh, uint32_t stride,
    int w_scaled, int h_scaled,
    int cells_x, int cells_y,
    double rotation_angle,
    uint8_t detail, uint8_t threshold_steps,
    uint8_t* modes,
    uint8_t* patterns,
    uint8_t* fg_colors,
    uint8_t* bg_colors) {
  
  // Calculate rotated dimensions
  double rad = rotation_angle * M_PI / 180.0;
  double cos_a = cos(fabs(rad));
  double sin_a = sin(fabs(rad));
  int rot_width = (int)ceil(fw * cos_a + fh * sin_a);
  int rot_height = (int)ceil(fw * sin_a + fh * cos_a);
  if (fmod(fabs(rotation_angle), 360.0) == 0.0) {
    rot_width = fw;
    rot_height = fh;
  }
  
  // Allocate device memory
  uint8_t *d_frame, *d_modes, *d_patterns, *d_fg_colors, *d_bg_colors;
  size_t frame_size = fh * stride;
  size_t output_size = cells_x * cells_y;
  
  cudaMalloc(&d_frame, frame_size);
  cudaMalloc(&d_modes, output_size);
  cudaMalloc(&d_patterns, output_size);
  cudaMalloc(&d_fg_colors, output_size * 3);
  cudaMalloc(&d_bg_colors, output_size * 3);
  
  // Copy input to device
  cudaMemcpy(d_frame, frame, frame_size, cudaMemcpyHostToDevice);
  
  // Launch kernel
  dim3 block(16, 16);
  dim3 grid((cells_x + block.x - 1) / block.x, 
            (cells_y + block.y - 1) / block.y);
  
  render_hybrid_kernel<<<grid, block>>>(
    d_frame, fw, fh, stride,
    w_scaled, h_scaled, cells_x, cells_y,
    rot_width, rot_height, rotation_angle,
    detail, threshold_steps,
    d_modes, d_patterns, d_fg_colors, d_bg_colors
  );
  
  // Copy results back
  cudaMemcpy(modes, d_modes, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(patterns, d_patterns, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(fg_colors, d_fg_colors, output_size * 3, cudaMemcpyDeviceToHost);
  cudaMemcpy(bg_colors, d_bg_colors, output_size * 3, cudaMemcpyDeviceToHost);
  
  // Free device memory
  cudaFree(d_frame);
  // Continuation of cuda_renderers.cu - complete the file

  cudaFree(d_modes);
  cudaFree(d_patterns);
  cudaFree(d_fg_colors);
  cudaFree(d_bg_colors);
}

// ============================================================================
// CUDA KERNEL: BLOCK RENDERING
// ============================================================================

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
    uint8_t* fg_colors,
    uint8_t* bg_colors) {
  
  int cx = blockIdx.x * blockDim.x + threadIdx.x;
  int cy = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (cx >= cells_x || cy >= cells_y) return;
  
  int idx = cy * cells_x + cx;
  
  // Each cell maps to 1 column × 2 rows in the rotated space
  int rot_x = cx * rot_width / w_scaled;
  int rot_y_top = (cy * 2) * rot_height / h_scaled;
  int rot_y_bot = (cy * 2 + 1) * rot_height / h_scaled;
  
  // Clamp coordinates in rotated space
  rot_x = min(max(rot_x, 0), rot_width - 1);
  rot_y_top = min(max(rot_y_top, 0), rot_height - 1);
  rot_y_bot = min(max(rot_y_bot, 0), rot_height - 1);
  
  // Sample top half with rotation
  uint8_t top_r, top_g, top_b;
  sample_rotated_pixel_cuda(frame_data, frame_width, frame_height, frame_stride,
                            rot_x, rot_y_top, rot_width, rot_height, rotation_angle, top_r, top_g, top_b);
  
  // Sample bottom half with rotation
  uint8_t bot_r, bot_g, bot_b;
  sample_rotated_pixel_cuda(frame_data, frame_width, frame_height, frame_stride,
                            rot_x, rot_y_bot, rot_width, rot_height, rotation_angle, bot_r, bot_g, bot_b);
  
  // Write output (fg = top, bg = bottom)
  fg_colors[idx * 3 + 0] = top_r;
  fg_colors[idx * 3 + 1] = top_g;
  fg_colors[idx * 3 + 2] = top_b;
  bg_colors[idx * 3 + 0] = bot_r;
  bg_colors[idx * 3 + 1] = bot_g;
  bg_colors[idx * 3 + 2] = bot_b;
}

// ============================================================================
// CUDA KERNEL: ASCII RENDERING
// ============================================================================

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
    uint8_t* intensities,
    uint8_t* fg_colors,
    uint8_t* bg_colors) {
  
  int cx = blockIdx.x * blockDim.x + threadIdx.x;
  int cy = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (cx >= cells_x || cy >= cells_y) return;
  
  int idx = cy * cells_x + cx;
  
  // Sample the cell area in rotated space
  int rot_x = cx * rot_width / w_scaled;
  int rot_y = cy * rot_height / h_scaled;
  
  // Clamp coordinates
  rot_x = min(max(rot_x, 0), rot_width - 1);
  rot_y = min(max(rot_y, 0), rot_height - 1);
  
  uint8_t r, g, b;
  sample_rotated_pixel_cuda(frame_data, frame_width, frame_height, frame_stride,
                            rot_x, rot_y, rot_width, rot_height, rotation_angle, r, g, b);
  
  double luma = luma_cuda(r, g, b);
  
  // Store intensity (0-255) and color
  intensities[idx] = (uint8_t)luma;
  fg_colors[idx * 3 + 0] = r;
  fg_colors[idx * 3 + 1] = g;
  fg_colors[idx * 3 + 2] = b;
  bg_colors[idx * 3 + 0] = 0;
  bg_colors[idx * 3 + 1] = 0;
  bg_colors[idx * 3 + 2] = 0;
}

// ============================================================================
// HOST WRAPPER: BLOCK RENDERING
// ============================================================================

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
    uint8_t* fg_colors,
    uint8_t* bg_colors) {
  
  // Calculate rotated dimensions
  double rad = rotation_angle * M_PI / 180.0;
  double cos_a = cos(fabs(rad));
  double sin_a = sin(fabs(rad));
  int rot_width = (int)ceil(frame_width * cos_a + frame_height * sin_a);
  int rot_height = (int)ceil(frame_width * sin_a + frame_height * cos_a);
  if (fmod(fabs(rotation_angle), 360.0) == 0.0) {
    rot_width = frame_width;
    rot_height = frame_height;
  }
  
  // Allocate device memory
  uint8_t *d_frame, *d_fg_colors, *d_bg_colors;
  size_t frame_size = frame_height * frame_stride;
  size_t output_size = cells_x * cells_y;
  
  cudaMalloc(&d_frame, frame_size);
  cudaMalloc(&d_fg_colors, output_size * 3);
  cudaMalloc(&d_bg_colors, output_size * 3);
  
  // Copy input to device
  cudaMemcpy(d_frame, frame_data, frame_size, cudaMemcpyHostToDevice);
  
  // Launch kernel
  dim3 block(16, 16);
  dim3 grid((cells_x + block.x - 1) / block.x, 
            (cells_y + block.y - 1) / block.y);
  
  render_blocks_kernel<<<grid, block>>>(
    d_frame, frame_width, frame_height, frame_stride,
    w_scaled, h_scaled, cells_x, cells_y,
    rot_width, rot_height, rotation_angle,
    d_fg_colors, d_bg_colors
  );
  
  // Copy results back
  cudaMemcpy(fg_colors, d_fg_colors, output_size * 3, cudaMemcpyDeviceToHost);
  cudaMemcpy(bg_colors, d_bg_colors, output_size * 3, cudaMemcpyDeviceToHost);
  
  // Free device memory
  cudaFree(d_frame);
  cudaFree(d_fg_colors);
  cudaFree(d_bg_colors);
}

// ============================================================================
// HOST WRAPPER: ASCII RENDERING
// ============================================================================

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
    uint8_t* intensities,
    uint8_t* fg_colors,
    uint8_t* bg_colors) {
  
  // Calculate rotated dimensions
  double rad = rotation_angle * M_PI / 180.0;
  double cos_a = cos(fabs(rad));
  double sin_a = sin(fabs(rad));
  int rot_width = (int)ceil(frame_width * cos_a + frame_height * sin_a);
  int rot_height = (int)ceil(frame_width * sin_a + frame_height * cos_a);
  if (fmod(fabs(rotation_angle), 360.0) == 0.0) {
    rot_width = frame_width;
    rot_height = frame_height;
  }
  
  // Allocate device memory
  uint8_t *d_frame, *d_intensities, *d_fg_colors, *d_bg_colors;
  size_t frame_size = frame_height * frame_stride;
  size_t output_size = cells_x * cells_y;
  
  cudaMalloc(&d_frame, frame_size);
  cudaMalloc(&d_intensities, output_size);
  cudaMalloc(&d_fg_colors, output_size * 3);
  cudaMalloc(&d_bg_colors, output_size * 3);
  
  // Copy input to device
  cudaMemcpy(d_frame, frame_data, frame_size, cudaMemcpyHostToDevice);
  
  // Launch kernel
  dim3 block(16, 16);
  dim3 grid((cells_x + block.x - 1) / block.x, 
            (cells_y + block.y - 1) / block.y);
  
  render_ascii_kernel<<<grid, block>>>(
    d_frame, frame_width, frame_height, frame_stride,
    w_scaled, h_scaled, cells_x, cells_y,
    rot_width, rot_height, rotation_angle,
    d_intensities, d_fg_colors, d_bg_colors
  );
  
  // Copy results back
  cudaMemcpy(intensities, d_intensities, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(fg_colors, d_fg_colors, output_size * 3, cudaMemcpyDeviceToHost);
  cudaMemcpy(bg_colors, d_bg_colors, output_size * 3, cudaMemcpyDeviceToHost);
  
  // Free device memory
  cudaFree(d_frame);
  cudaFree(d_intensities);
  cudaFree(d_fg_colors);
  cudaFree(d_bg_colors);
}

// ============================================================================
// ERROR CHECKING UTILITIES
// ============================================================================

extern "C" bool check_cuda_availability() {
  int deviceCount = 0;
  cudaError_t error = cudaGetDeviceCount(&deviceCount);
  
  if (error != cudaSuccess || deviceCount == 0) {
    return false;
  }
  
  return true;
}

extern "C" void print_cuda_device_info() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  
  if (deviceCount == 0) {
    fprintf(stderr, "[CUDA] No CUDA devices found\n");
    return;
  }
  
  for (int dev = 0; dev < deviceCount; dev++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    
    fprintf(stderr, "[CUDA] Device %d: %s\n", dev, prop.name);
    fprintf(stderr, "  Compute Capability: %d.%d\n", prop.major, prop.minor);
    fprintf(stderr, "  Total Global Memory: %.2f GB\n", 
            prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    fprintf(stderr, "  Multiprocessors: %d\n", prop.multiProcessorCount);
    fprintf(stderr, "  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    fprintf(stderr, "  Max Grid Size: [%d, %d, %d]\n", 
            prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  }
}