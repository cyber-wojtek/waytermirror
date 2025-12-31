#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <vector>

// CUDA kernel for braille pattern calculation
__device__ uint8_t calculate_braille_pattern_device(
    const double* lumas,
    const uint8_t* colors,  // Flattened: [r0,g0,b0, r1,g1,b1, ...]
    uint8_t detail_level,
    uint8_t quality) {
  
  // Quality determines search step size: quality 0=step 16, quality 50=step 4, quality 100=step 1
  int step = max(1, 16 - (quality / 7));
  
  // HIGH DETAIL (>= 90): Effective luminance with saturation boost
  if (detail_level >= 90) {
    double eff_luma[8];
    for (int i = 0; i < 8; i++) {
      uint8_t r = colors[i * 3 + 0];
      uint8_t g = colors[i * 3 + 1];
      uint8_t b = colors[i * 3 + 2];
      
      // HSL saturation
      uint8_t maxc = (r > g) ? (r > b ? r : b) : (g > b ? g : b);
      uint8_t minc = (r < g) ? (r < b ? r : b) : (g < b ? g : b);
      double sat = (maxc == minc) ? 0.0 : (double)(maxc - minc) / maxc;
      
      // Effective luminance: luma + saturation boost
      double base_luma = 0.299 * r + 0.587 * g + 0.114 * b;
      eff_luma[i] = base_luma + sat * 60.0;  // Boost saturated colors by up to 60 luma units
    }
    
    // Otsu threshold on effective luminance
    double best_threshold = 127;
    double best_separation = 0;
    
    for (int t_int = 0; t_int <= 255; t_int += step) {
      double t = (double)t_int;
      double sum_below = 0, sum_above = 0;
      int count_below = 0, count_above = 0;
      
      for (int i = 0; i < 8; i++) {
        if (eff_luma[i] < t) {
          sum_below += eff_luma[i];
          count_below++;
        } else {
          sum_above += eff_luma[i];
          count_above++;
        }
      }
      
      if (count_below > 0 && count_above > 0) {
        double mean_below = sum_below / count_below;
        double mean_above = sum_above / count_above;
        double separation = count_below * count_above * (mean_above - mean_below) * (mean_above - mean_below);
        
        if (separation > best_separation) {
          best_separation = separation;
          best_threshold = t;
        }
      }
    }
    
    // Apply threshold
    uint8_t pattern = 0;
    for (int dot = 0; dot < 8; dot++) {
      if (eff_luma[dot] > best_threshold) {
        pattern |= (1 << dot);
      }
    }
    return pattern;
  }
  
  // MEDIUM-HIGH DETAIL (70-89): Otsu on standard luminance
  if (detail_level >= 70) {
    double best_threshold = 127;
    double best_separation = 0;
    
    for (int t_int = 0; t_int <= 255; t_int += step) {
      double t = (double)t_int;
      double sum_below = 0, sum_above = 0;
      int count_below = 0, count_above = 0;
      
      for (int i = 0; i < 8; i++) {
        if (lumas[i] < t) {
          sum_below += lumas[i];
          count_below++;
        } else {
          sum_above += lumas[i];
          count_above++;
        }
      }
      
      if (count_below > 0 && count_above > 0) {
        double mean_below = sum_below / count_below;
        double mean_above = sum_above / count_above;
        double separation = count_below * count_above * (mean_above - mean_below) * (mean_above - mean_below);
        
        if (separation > best_separation) {
          best_separation = separation;
          best_threshold = t;
        }
      }
    }
    
    // Apply threshold
    uint8_t pattern = 0;
    for (int dot = 0; dot < 8; dot++) {
      if (lumas[dot] > best_threshold) {
        pattern |= (1 << dot);
      }
    }
    return pattern;
  }
  
  // LOW-MEDIUM DETAIL (<70): Simple mean threshold (matches CPU)
  double mean_luma = 0;
  for (int i = 0; i < 8; i++) {
    mean_luma += lumas[i];
  }
  mean_luma /= 8.0;
  
  double threshold = (detail_level >= 40) ? mean_luma : (mean_luma + 10.0);
  
  uint8_t pattern = 0;
  for (int dot = 0; dot < 8; dot++) {
    double diff = lumas[dot] - threshold;
    bool activate;
    if (detail_level < 50) {
      activate = (diff > 6.0) || (diff > -6.0 && diff > 0);
    } else {
      activate = (diff >= 0);
    }
    if (activate) {
      pattern |= (1 << dot);
    }
  }
  
  return pattern;
}

__device__ void calculate_braille_colors_device(
    const uint8_t* colors,
    uint8_t pattern,
    uint8_t* fg_color,
    uint8_t* bg_color,
    uint8_t detail_level) {
  
  int fg_dots[8], bg_dots[8];
  int fg_count = 0, bg_count = 0;
  
  for (int dot = 0; dot < 8; dot++) {
    if (pattern & (1 << dot)) {
      fg_dots[fg_count++] = dot;
    } else {
      bg_dots[bg_count++] = dot;
    }
  }
  
  // If degenerate, just average all
  if (fg_count == 0 || bg_count == 0) {
    uint32_t r = 0, g = 0, b = 0;
    for (int dot = 0; dot < 8; dot++) {
      r += colors[dot * 3 + 0];
      g += colors[dot * 3 + 1];
      b += colors[dot * 3 + 2];
    }
    fg_color[0] = bg_color[0] = r / 8;
    fg_color[1] = bg_color[1] = g / 8;
    fg_color[2] = bg_color[2] = b / 8;
    return;
  }
  
  // Average colors
  uint32_t fg_r = 0, fg_g = 0, fg_b = 0;
  uint32_t bg_r = 0, bg_g = 0, bg_b = 0;
  
  for (int i = 0; i < fg_count; i++) {
    int dot = fg_dots[i];
    fg_r += colors[dot * 3 + 0];
    fg_g += colors[dot * 3 + 1];
    fg_b += colors[dot * 3 + 2];
  }
  
  for (int i = 0; i < bg_count; i++) {
    int dot = bg_dots[i];
    bg_r += colors[dot * 3 + 0];
    bg_g += colors[dot * 3 + 1];
    bg_b += colors[dot * 3 + 2];
  }
  
  fg_color[0] = fg_r / fg_count;
  fg_color[1] = fg_g / fg_count;
  fg_color[2] = fg_b / fg_count;
  
  bg_color[0] = bg_r / bg_count;
  bg_color[1] = bg_g / bg_count;
  bg_color[2] = bg_b / bg_count;
  
  // For low detail, check if luminance inversion needed (match CPU)
  if (detail_level < 70) {
    double fg_luma = 0.299 * fg_color[0] + 0.587 * fg_color[1] + 0.114 * fg_color[2];
    double bg_luma = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2];
    
    if (fg_luma < bg_luma - 15.0) {
      // Swap fg and bg
      uint8_t tmp[3];
      tmp[0] = fg_color[0]; tmp[1] = fg_color[1]; tmp[2] = fg_color[2];
      fg_color[0] = bg_color[0]; fg_color[1] = bg_color[1]; fg_color[2] = bg_color[2];
      bg_color[0] = tmp[0]; bg_color[1] = tmp[1]; bg_color[2] = tmp[2];
    }
  }
}

// Main CUDA kernel
__global__ void render_hybrid_kernel(
    const uint8_t* frame,
    uint32_t fw, uint32_t fh, uint32_t stride,
    int w_scaled, int h_scaled,
    int cells_x, int cells_y,
    uint8_t detail,
    uint8_t quality,
    uint8_t* modes,
    uint8_t* patterns,
    uint8_t* fg,
    uint8_t* bg)
{
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;
    if (cx >= cells_x || cy >= cells_y) return;

    int idx = cy * cells_x + cx;

    double luma[8];
    uint8_t col[24];

    const int dx[8] = {0,0,0,1,1,1,0,1};
    const int dy[8] = {0,1,2,0,1,2,3,3};

    double mean = 0.0;

    for (int d = 0; d < 8; d++) {
        double fx = (cx*2 + dx[d] + 0.5) / w_scaled;
        double fy = (cy*4 + dy[d] + 0.5) / h_scaled;

        int x = min(max(int(fx * fw), 0), int(fw-1));
        int y = min(max(int(fy * fh), 0), int(fh-1));

        const uint8_t* p = frame + y*stride + x*4;
        uint8_t r = p[2], g = p[1], b = p[0];

        col[d*3+0] = r;
        col[d*3+1] = g;
        col[d*3+2] = b;

        luma[d] = 0.299*r + 0.587*g + 0.114*b;
        mean += luma[d];
    }

    mean /= 8.0;

    double var = 0.0;
    for (int i=0;i<8;i++){
        double d = luma[i]-mean;
        var += d*d;
    }
    var /= 8.0;

    double var_thresh = 1000.0 - (detail/100.0)*950.0;

    // ──────────────── BRAILLE PATH ────────────────
    if (var > var_thresh) {
        uint8_t pat = calculate_braille_pattern_device(
            luma, col, detail, quality);

        uint8_t fg_c[3], bg_c[3];
        calculate_braille_colors_device(col, pat, fg_c, bg_c, detail);

        modes[idx]    = 1;
        patterns[idx] = pat;

        for(int i=0;i<3;i++){
            fg[idx*3+i] = fg_c[i];
            bg[idx*3+i] = bg_c[i];
        }
        return;
    }

    // ──────────────── HALF BLOCK PATH ────────────────
    int sy0 = min(int((cy*4.0)   * fh / h_scaled), int(fh-1));
    int sy1 = min(int((cy*4.0+2) * fh / h_scaled), int(fh-1));
    int sx  = min(int((cx*2.0)   * fw / w_scaled), int(fw-1));

    uint32_t tr=0,tg=0,tb=0, br=0,bg_=0,bb=0;

    for(int ky=0;ky<2;ky++){
        for(int kx=0;kx<2;kx++){
            int px = min(sx+kx, int(fw-1));

            const uint8_t* pt = frame + min(sy0+ky, int(fh-1))*stride + px*4;
            const uint8_t* pb = frame + min(sy1+ky, int(fh-1))*stride + px*4;

            tr+=pt[2]; tg+=pt[1]; tb+=pt[0];
            br+=pb[2]; bg_+=pb[1]; bb+=pb[0];
        }
    }

    modes[idx] = 0;
    patterns[idx] = 0;

    fg[idx*3+0] = tr/4; fg[idx*3+1] = tg/4; fg[idx*3+2] = tb/4;
    bg[idx*3+0] = br/4; bg[idx*3+1] = bg_/4; bg[idx*3+2] = bb/4;
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
    uint8_t detail_level,
    uint8_t quality,
    uint8_t* patterns,      // Output: braille patterns
    uint8_t* fg_colors,     // Output: foreground colors (RGB)
    uint8_t* bg_colors) {   // Output: background colors (RGB)
  
  int cx = blockIdx.x * blockDim.x + threadIdx.x;
  int cy = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (cx >= cells_x || cy >= cells_y) return;
  
  int cell_idx = cy * cells_x + cx;
  
  // Analyze braille cell (sample 8 dots)
  double lumas[8];
  uint8_t colors[24];  // 8 dots * 3 channels
  
  const int dot_cols[8] = {0, 0, 0, 1, 1, 1, 0, 1};
  const int dot_rows[8] = {0, 1, 2, 0, 1, 2, 3, 3};
  
  for (int dot = 0; dot < 8; dot++) {
    double dot_x_frac = (cx * 2.0 + dot_cols[dot] + 0.5) / w_scaled;
    double dot_y_frac = (cy * 4.0 + dot_rows[dot] + 0.5) / h_scaled;
    
    int src_x = (int)(dot_x_frac * frame_width);
    int src_y = (int)(dot_y_frac * frame_height);
    
    src_x = min(max(src_x, 0), (int)frame_width - 1);
    src_y = min(max(src_y, 0), (int)frame_height - 1);
    
    // Simple 2x2 average
    uint32_t r_sum = 0, g_sum = 0, b_sum = 0;
    int count = 0;
    
    for (int ky = 0; ky < 2; ky++) {
      for (int kx = 0; kx < 2; kx++) {
        int px = min(max(src_x + kx, 0), (int)frame_width - 1);
        int py = min(max(src_y + ky, 0), (int)frame_height - 1);
        
        const uint8_t* p = frame_data + py * frame_stride + px * 4;
        uint8_t b = p[0];
        uint8_t g = p[1];
        uint8_t r = p[2];
        
        r_sum += r;
        g_sum += g;
        b_sum += b;
        count++;
      }
    }
    
    uint8_t r = r_sum / count;
    uint8_t g = g_sum / count;
    uint8_t b = b_sum / count;
    
    colors[dot * 3 + 0] = r;
    colors[dot * 3 + 1] = g;
    colors[dot * 3 + 2] = b;
    
    lumas[dot] = 0.299 * r + 0.587 * g + 0.114 * b;
  }
  
  // Calculate pattern
  uint8_t pattern = calculate_braille_pattern_device(lumas, colors, detail_level, quality);
  patterns[cell_idx] = pattern;
  
  // Calculate colors
  uint8_t fg_color[3], bg_color[3];
  calculate_braille_colors_device(colors, pattern, fg_color, bg_color, detail_level);
  
  fg_colors[cell_idx * 3 + 0] = fg_color[0];
  fg_colors[cell_idx * 3 + 1] = fg_color[1];
  fg_colors[cell_idx * 3 + 2] = fg_color[2];
  
  bg_colors[cell_idx * 3 + 0] = bg_color[0];
  bg_colors[cell_idx * 3 + 1] = bg_color[1];
  bg_colors[cell_idx * 3 + 2] = bg_color[2];
}

// Host wrapper function
extern "C" void render_hybrid_cuda(
    const uint8_t* frame,
    uint32_t fw, uint32_t fh, uint32_t stride,
    int w_scaled, int h_scaled,
    int cells_x, int cells_y,
    uint8_t detail, uint8_t quality,
    uint8_t* modes,
    uint8_t* patterns,
    uint8_t* fg,
    uint8_t* bg)
{
    size_t frame_sz = fh * stride;
    size_t cells = cells_x * cells_y;

    uint8_t *d_frame, *d_modes, *d_patterns, *d_fg, *d_bg;

    cudaMalloc(&d_frame, frame_sz);
    cudaMalloc(&d_modes, cells);
    cudaMalloc(&d_patterns, cells);
    cudaMalloc(&d_fg, cells*3);
    cudaMalloc(&d_bg, cells*3);

    cudaMemcpy(d_frame, frame, frame_sz, cudaMemcpyHostToDevice);

    dim3 blk(16,16);
    dim3 grd((cells_x+15)/16, (cells_y+15)/16);

    render_hybrid_kernel<<<grd,blk>>>(
        d_frame, fw, fh, stride,
        w_scaled, h_scaled,
        cells_x, cells_y,
        detail, quality,
        d_modes, d_patterns, d_fg, d_bg);

    cudaDeviceSynchronize();

    cudaMemcpy(modes, d_modes, cells, cudaMemcpyDeviceToHost);
    cudaMemcpy(patterns, d_patterns, cells, cudaMemcpyDeviceToHost);
    cudaMemcpy(fg, d_fg, cells*3, cudaMemcpyDeviceToHost);
    cudaMemcpy(bg, d_bg, cells*3, cudaMemcpyDeviceToHost);

    cudaFree(d_frame);
    cudaFree(d_modes);
    cudaFree(d_patterns);
    cudaFree(d_fg);
    cudaFree(d_bg);
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
    uint8_t quality,
    uint8_t* patterns,
    uint8_t* fg_colors,
    uint8_t* bg_colors) {
  
  // Allocate device memory
  uint8_t *d_frame, *d_patterns, *d_fg_colors, *d_bg_colors;
  
  size_t frame_size = frame_height * frame_stride;
  size_t cell_count = cells_x * cells_y;
  
  cudaMalloc(&d_frame, frame_size);
  cudaMalloc(&d_patterns, cell_count);
  cudaMalloc(&d_fg_colors, cell_count * 3);
  cudaMalloc(&d_bg_colors, cell_count * 3);
  
  // Copy frame to device
  cudaMemcpy(d_frame, frame_data, frame_size, cudaMemcpyHostToDevice);
  
  // Launch kernel
  dim3 block(16, 16);
  dim3 grid((cells_x + block.x - 1) / block.x, (cells_y + block.y - 1) / block.y);
  
  render_braille_kernel<<<grid, block>>>(
    d_frame, frame_width, frame_height, frame_stride,
    w_scaled, h_scaled, cells_x, cells_y,
    detail_level, quality,
    d_patterns, d_fg_colors, d_bg_colors
  );
  
  cudaDeviceSynchronize();
  
  // Copy results back
  cudaMemcpy(patterns, d_patterns, cell_count, cudaMemcpyDeviceToHost);
  cudaMemcpy(fg_colors, d_fg_colors, cell_count * 3, cudaMemcpyDeviceToHost);
  cudaMemcpy(bg_colors, d_bg_colors, cell_count * 3, cudaMemcpyDeviceToHost);
  
  // Free device memory
  cudaFree(d_frame);
  cudaFree(d_patterns);
  cudaFree(d_fg_colors);
  cudaFree(d_bg_colors);
}