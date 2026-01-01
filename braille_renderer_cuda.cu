#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

// ============================================================================
// CUDA KERNEL HELPERS
// ============================================================================

__device__ inline void get_rgb(const uint8_t *p, uint8_t &r, uint8_t &g, uint8_t &b)
{
  b = p[0];
  g = p[1];
  r = p[2];
}

__device__ inline double calculate_luma(uint8_t r, uint8_t g, uint8_t b)
{
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

__device__ inline double calculate_effective_luma(uint8_t r, uint8_t g, uint8_t b)
{
  uint8_t maxc = max(max(r, g), b);
  uint8_t minc = min(min(r, g), b);
  double sat = (maxc == minc) ? 0.0 : (double)(maxc - minc) / maxc;
  double base_luma = calculate_luma(r, g, b);
  return base_luma + sat * 60.0;
}

// ============================================================================
// REGIONAL COLOR ANALYSIS (for braille)
// ============================================================================

__device__ void analyze_regional_colors_device(
    const uint8_t *frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int cell_x,
    int cell_y,
    int w_scaled,
    int h_scaled,
    int cells_x,
    int cells_y,
    uint8_t detail_level,
    uint8_t &dominant_fg_r, uint8_t &dominant_fg_g, uint8_t &dominant_fg_b,
    uint8_t &dominant_bg_r, uint8_t &dominant_bg_g, uint8_t &dominant_bg_b)
{
  int region_size = 5;
  if (detail_level >= 90)
    region_size = 3;
  else if (detail_level < 50)
    region_size = 7;

  uint32_t reds[200], greens[200], blues[200];
  uint8_t lumas[200];
  int count = 0;

  int half_region = region_size / 2;

  for (int dy = -half_region; dy <= half_region && count < 200; dy++)
  {
    for (int dx = -half_region; dx <= half_region && count < 200; dx++)
    {
      int nx = cell_x + dx;
      int ny = cell_y + dy;

      if (nx < 0 || nx >= cells_x || ny < 0 || ny >= cells_y)
        continue;

      // Sample 8 dots
      int dot_positions[8][2] = {
          {0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {0, 3}, {1, 3}};

      for (int dot = 0; dot < 8 && count < 200; dot++)
      {
        int dot_x = dot_positions[dot][0];
        int dot_y = dot_positions[dot][1];

        int src_x = (nx * 2 + dot_x) * frame_width / w_scaled;
        int src_y = (ny * 4 + dot_y) * frame_height / h_scaled;

        src_x = min(max(src_x, 0), (int)frame_width - 1);
        src_y = min(max(src_y, 0), (int)frame_height - 1);

        const uint8_t *p = frame_data + src_y * frame_stride + src_x * 4;
        uint8_t r, g, b;
        get_rgb(p, r, g, b);

        reds[count] = r;
        greens[count] = g;
        blues[count] = b;
        lumas[count] = (uint8_t)calculate_luma(r, g, b);
        count++;
      }
    }
  }

  if (count == 0)
  {
    dominant_fg_r = dominant_fg_g = dominant_fg_b = 128;
    dominant_bg_r = dominant_bg_g = dominant_bg_b = 64;
    return;
  }

  // Find median luma
  uint8_t sorted_lumas[200];
  for (int i = 0; i < count; i++)
    sorted_lumas[i] = lumas[i];

  // Simple insertion sort (fast for small arrays)
  for (int i = 1; i < count; i++)
  {
    uint8_t key = sorted_lumas[i];
    int j = i - 1;
    while (j >= 0 && sorted_lumas[j] > key)
    {
      sorted_lumas[j + 1] = sorted_lumas[j];
      j--;
    }
    sorted_lumas[j + 1] = key;
  }

  uint8_t median_luma = sorted_lumas[count / 2];

  // Split into light/dark groups
  uint32_t light_r = 0, light_g = 0, light_b = 0, light_count = 0;
  uint32_t dark_r = 0, dark_g = 0, dark_b = 0, dark_count = 0;

  for (int i = 0; i < count; i++)
  {
    if (lumas[i] >= median_luma)
    {
      light_r += reds[i];
      light_g += greens[i];
      light_b += blues[i];
      light_count++;
    }
    else
    {
      dark_r += reds[i];
      dark_g += greens[i];
      dark_b += blues[i];
      dark_count++;
    }
  }

  uint8_t light_avg_r = light_count > 0 ? light_r / light_count : 200;
  uint8_t light_avg_g = light_count > 0 ? light_g / light_count : 200;
  uint8_t light_avg_b = light_count > 0 ? light_b / light_count : 200;

  uint8_t dark_avg_r = dark_count > 0 ? dark_r / dark_count : 55;
  uint8_t dark_avg_g = dark_count > 0 ? dark_g / dark_count : 55;
  uint8_t dark_avg_b = dark_count > 0 ? dark_b / dark_count : 55;

  // Majority is background, minority is foreground
  if (light_count >= dark_count)
  {
    dominant_bg_r = light_avg_r;
    dominant_bg_g = light_avg_g;
    dominant_bg_b = light_avg_b;
    dominant_fg_r = dark_avg_r;
    dominant_fg_g = dark_avg_g;
    dominant_fg_b = dark_avg_b;
  }
  else
  {
    dominant_bg_r = dark_avg_r;
    dominant_bg_g = dark_avg_g;
    dominant_bg_b = dark_avg_b;
    dominant_fg_r = light_avg_r;
    dominant_fg_g = light_avg_g;
    dominant_fg_b = light_avg_b;
  }
}

// ============================================================================
// BRAILLE CELL ANALYSIS
// ============================================================================

__device__ void analyze_braille_cell_device(
    const uint8_t *frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int cell_x,
    int cell_y,
    int w_scaled,
    int h_scaled,
    uint8_t detail_level,
    double lumas[8],
    uint8_t colors[8][3],
    bool &has_edge,
    double &edge_strength,
    double &mean_luma)
{
  int kernel_size;
  if (detail_level >= 95)
    kernel_size = 1;
  else if (detail_level >= 80)
    kernel_size = 2;
  else if (detail_level >= 60)
    kernel_size = 3;
  else if (detail_level >= 40)
    kernel_size = 4;
  else
    kernel_size = 2;

  int dot_positions[8][2] = {
      {0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {0, 3}, {1, 3}};

  // Sample each dot with kernel
  for (int dot = 0; dot < 8; dot++)
  {
    int dot_x = dot_positions[dot][0];
    int dot_y = dot_positions[dot][1];

    int src_x = (cell_x * 2 + dot_x) * frame_width / w_scaled;
    int src_y = (cell_y * 4 + dot_y) * frame_height / h_scaled;

    uint32_t r_sum = 0, g_sum = 0, b_sum = 0;
    double weight_sum = 0;

    for (int ky = 0; ky < kernel_size; ky++)
    {
      for (int kx = 0; kx < kernel_size; kx++)
      {
        int px = min(max(src_x + kx - kernel_size / 2, 0), (int)frame_width - 1);
        int py = min(max(src_y + ky - kernel_size / 2, 0), (int)frame_height - 1);

        const uint8_t *p = frame_data + py * frame_stride + px * 4;
        uint8_t r, g, b;
        get_rgb(p, r, g, b);

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

    colors[dot][0] = r;
    colors[dot][1] = g;
    colors[dot][2] = b;

    lumas[dot] = calculate_luma(r, g, b);
  }

  // Edge detection
  int adjacency_pairs[][2] = {
      {0, 1}, {1, 2}, {3, 4}, {4, 5}, {6, 7}, // Vertical
      {0, 3},
      {1, 4},
      {2, 5},
      {6, 7}, // Horizontal
      {0, 4},
      {1, 3},
      {1, 5},
      {2, 4} // Diagonal
  };

  has_edge = false;
  double max_edge = 0;
  double edge_threshold = (detail_level >= 70) ? 30.0 : 50.0;

  for (int i = 0; i < 13; i++)
  {
    int idx1 = adjacency_pairs[i][0];
    int idx2 = adjacency_pairs[i][1];

    double dr = colors[idx1][0] - colors[idx2][0];
    double dg = colors[idx1][1] - colors[idx2][1];
    double db = colors[idx1][2] - colors[idx2][2];
    double color_dist = sqrt(2.0 * dr * dr + 4.0 * dg * dg + 3.0 * db * db);

    double luma_diff = fabs(lumas[idx1] - lumas[idx2]);
    double edge = fmax(color_dist, luma_diff * 2.0);

    max_edge = fmax(max_edge, edge);

    if (edge > edge_threshold)
    {
      has_edge = true;
    }
  }

  edge_strength = max_edge;

  // Calculate mean luma
  double sum = 0;
  for (int i = 0; i < 8; i++)
  {
    sum += lumas[i];
  }
  mean_luma = sum / 8.0;
}

// ============================================================================
// BRAILLE PATTERN CALCULATION
// ============================================================================

__device__ uint8_t calculate_braille_pattern_device(
    const double lumas[8],
    const uint8_t colors[8][3],
    uint8_t detail_level,
    uint8_t quality,
    double mean_luma,
    uint8_t regional_bg_r,
    uint8_t regional_bg_g,
    uint8_t regional_bg_b)
{
  int step = max(1, 16 - (quality / 7));

  uint8_t pattern = 0;
  double threshold;

  // HIGH DETAIL: Otsu on effective luminance
  if (detail_level >= 90)
  {
    double eff_luma[8];
    for (int i = 0; i < 8; i++)
    {
      uint8_t r = colors[i][0];
      uint8_t g = colors[i][1];
      uint8_t b = colors[i][2];

      uint8_t maxc = max(max(r, g), b);
      uint8_t minc = min(min(r, g), b);
      double sat = (maxc == minc) ? 0.0 : (double)(maxc - minc) / maxc;

      double base_luma = calculate_luma(r, g, b);
      eff_luma[i] = base_luma + sat * 60.0;
    }

    double best_threshold = 127;
    double best_separation = 0;

    for (int t_int = 0; t_int <= 255; t_int += step)
    {
      double t = t_int;
      double sum_below = 0, sum_above = 0;
      int count_below = 0, count_above = 0;

      for (int i = 0; i < 8; i++)
      {
        if (eff_luma[i] < t)
        {
          sum_below += eff_luma[i];
          count_below++;
        }
        else
        {
          sum_above += eff_luma[i];
          count_above++;
        }
      }

      if (count_below > 0 && count_above > 0)
      {
        double mean_below = sum_below / count_below;
        double mean_above = sum_above / count_above;
        double separation = count_below * count_above *
                            (mean_above - mean_below) * (mean_above - mean_below);

        if (separation > best_separation)
        {
          best_separation = separation;
          best_threshold = t;
        }
      }
    }

    for (int dot = 0; dot < 8; dot++)
    {
      if (eff_luma[dot] > best_threshold)
      {
        pattern |= (1 << dot);
      }
    }
  }
  // MEDIUM-HIGH DETAIL: Otsu on luminance
  else if (detail_level >= 70)
  {
    double best_threshold = mean_luma;
    double best_separation = 0;

    for (int t_int = 0; t_int <= 255; t_int += step)
    {
      double t = t_int;
      double sum_below = 0, sum_above = 0;
      int count_below = 0, count_above = 0;

      for (int i = 0; i < 8; i++)
      {
        if (lumas[i] < t)
        {
          sum_below += lumas[i];
          count_below++;
        }
        else
        {
          sum_above += lumas[i];
          count_above++;
        }
      }

      if (count_below > 0 && count_above > 0)
      {
        double mean_below = sum_below / count_below;
        double mean_above = sum_above / count_above;
        double local_contrast = mean_above - mean_below;
        double separation = count_below * count_above * local_contrast * local_contrast;

        if (separation > best_separation)
        {
          best_separation = separation;
          best_threshold = t;
        }
      }
    }
    threshold = best_threshold;

    for (int dot = 0; dot < 8; dot++)
    {
      if (lumas[dot] >= threshold)
      {
        pattern |= (1 << dot);
      }
    }
  }
  // LOWER DETAIL: Simple threshold
  else
  {
    threshold = (detail_level >= 40) ? mean_luma : mean_luma + 10.0;
    threshold = min(max(threshold, 0.0), 255.0);

    for (int dot = 0; dot < 8; dot++)
    {
      double diff = lumas[dot] - threshold;
      bool activate = (detail_level < 50) ? (diff > 6.0 || (diff > -6.0 && diff > 0)) : (diff >= 0);

      if (activate)
      {
        pattern |= (1 << dot);
      }
    }
  }

  // Check if we need to invert based on regional context
  uint32_t lit_r = 0, lit_g = 0, lit_b = 0, lit_count = 0;
  uint32_t unlit_r = 0, unlit_g = 0, unlit_b = 0, unlit_count = 0;

  for (int dot = 0; dot < 8; dot++)
  {
    if (pattern & (1 << dot))
    {
      lit_r += colors[dot][0];
      lit_g += colors[dot][1];
      lit_b += colors[dot][2];
      lit_count++;
    }
    else
    {
      unlit_r += colors[dot][0];
      unlit_g += colors[dot][1];
      unlit_b += colors[dot][2];
      unlit_count++;
    }
  }

  if (lit_count > 0 && unlit_count > 0)
  {
    double lit_avg_r = lit_r / (double)lit_count;
    double lit_avg_g = lit_g / (double)lit_count;
    double lit_avg_b = lit_b / (double)lit_count;

    double unlit_avg_r = unlit_r / (double)unlit_count;
    double unlit_avg_g = unlit_g / (double)unlit_count;
    double unlit_avg_b = unlit_b / (double)unlit_count;

    double lit_to_reg_bg =
        fabs(lit_avg_r - regional_bg_r) +
        fabs(lit_avg_g - regional_bg_g) +
        fabs(lit_avg_b - regional_bg_b);

    double unlit_to_reg_bg =
        fabs(unlit_avg_r - regional_bg_r) +
        fabs(unlit_avg_g - regional_bg_g) +
        fabs(unlit_avg_b - regional_bg_b);

    if (lit_to_reg_bg < unlit_to_reg_bg)
    {
      pattern ^= 0xFF;
    }
  }

  return pattern;
}

// ============================================================================
// BRAILLE COLOR CALCULATION
// ============================================================================

__device__ void calculate_braille_colors_device(
    const uint8_t colors[8][3],
    const double lumas[8],
    uint8_t pattern,
    uint8_t detail_level,
    uint8_t &fg_r, uint8_t &fg_g, uint8_t &fg_b,
    uint8_t &bg_r, uint8_t &bg_g, uint8_t &bg_b)
{
  uint32_t lit_r = 0, lit_g = 0, lit_b = 0, lit_count = 0;
  uint32_t unlit_r = 0, unlit_g = 0, unlit_b = 0, unlit_count = 0;

  for (int i = 0; i < 8; i++)
  {
    if (pattern & (1 << i))
    {
      lit_r += colors[i][0];
      lit_g += colors[i][1];
      lit_b += colors[i][2];
      lit_count++;
    }
    else
    {
      unlit_r += colors[i][0];
      unlit_g += colors[i][1];
      unlit_b += colors[i][2];
      unlit_count++;
    }
  }

  if (lit_count == 0)
  {
    fg_r = unlit_r / 8;
    fg_g = unlit_g / 8;
    fg_b = unlit_b / 8;
    bg_r = fg_r;
    bg_g = fg_g;
    bg_b = fg_b;
  }
  else if (unlit_count == 0)
  {
    fg_r = lit_r / 8;
    fg_g = lit_g / 8;
    fg_b = lit_b / 8;
    bg_r = fg_r;
    bg_g = fg_g;
    bg_b = fg_b;
  }
  else
  {
    fg_r = lit_r / lit_count;
    fg_g = lit_g / lit_count;
    fg_b = lit_b / lit_count;

    bg_r = unlit_r / unlit_count;
    bg_g = unlit_g / unlit_count;
    bg_b = unlit_b / unlit_count;
  }

  // Inversion check for lower detail
  if (detail_level < 70)
  {
    double fg_luma = calculate_luma(fg_r, fg_g, fg_b);
    double bg_luma = calculate_luma(bg_r, bg_g, bg_b);

    if (fg_luma < bg_luma - 15.0)
    {
      uint8_t tmp;
      tmp = fg_r;
      fg_r = bg_r;
      bg_r = tmp;
      tmp = fg_g;
      fg_g = bg_g;
      bg_g = tmp;
      tmp = fg_b;
      fg_b = bg_b;
      bg_b = tmp;
    }
  }
}

// ============================================================================
// CUDA KERNEL: BRAILLE RENDERER
// ============================================================================

__global__ void render_braille_kernel(
    const uint8_t *frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int w_scaled,
    int h_scaled,
    int cells_x,
    int cells_y,
    uint8_t detail_level,
    uint8_t threshold_steps,
    uint8_t *patterns,
    uint8_t *fg_colors,
    uint8_t *bg_colors)
{
  int cx = blockIdx.x * blockDim.x + threadIdx.x;
  int cy = blockIdx.y * blockDim.y + threadIdx.y;

  if (cx >= cells_x || cy >= cells_y)
    return;

  int idx = cy * cells_x + cx;

  // Analyze braille cell
  double lumas[8];
  uint8_t colors[8][3];
  bool has_edge;
  double edge_strength;
  double mean_luma;

  analyze_braille_cell_device(
      frame_data, frame_width, frame_height, frame_stride,
      cx, cy, w_scaled, h_scaled, detail_level,
      lumas, colors, has_edge, edge_strength, mean_luma);

  // Regional color analysis
  uint8_t regional_fg_r, regional_fg_g, regional_fg_b;
  uint8_t regional_bg_r, regional_bg_g, regional_bg_b;

  analyze_regional_colors_device(
      frame_data, frame_width, frame_height, frame_stride,
      cx, cy, w_scaled, h_scaled, cells_x, cells_y, detail_level,
      regional_fg_r, regional_fg_g, regional_fg_b,
      regional_bg_r, regional_bg_g, regional_bg_b);

  // Calculate pattern
  uint8_t pattern = calculate_braille_pattern_device(
      lumas, colors, detail_level, threshold_steps, mean_luma,
      regional_bg_r, regional_bg_g, regional_bg_b);

  // Calculate colors
  uint8_t fg_r, fg_g, fg_b, bg_r, bg_g, bg_b;
  calculate_braille_colors_device(
      colors, lumas, pattern, detail_level,
      fg_r, fg_g, fg_b, bg_r, bg_g, bg_b);

  // Write results
  patterns[idx] = pattern;
  fg_colors[idx * 3 + 0] = fg_r;
  fg_colors[idx * 3 + 1] = fg_g;
  fg_colors[idx * 3 + 2] = fg_b;
  bg_colors[idx * 3 + 0] = bg_r;
  bg_colors[idx * 3 + 1] = bg_g;
  bg_colors[idx * 3 + 2] = bg_b;
}

// ============================================================================
// CUDA KERNEL: HYBRID RENDERER
// ============================================================================

__global__ void render_hybrid_kernel(
    const uint8_t *frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int w_scaled,
    int h_scaled,
    int cells_x,
    int cells_y,
    uint8_t detail_level,
    uint8_t threshold_steps,
    uint8_t *modes, // 1=braille, 0=half-block
    uint8_t *patterns,
    uint8_t *fg_colors,
    uint8_t *bg_colors)
{
  int cx = blockIdx.x * blockDim.x + threadIdx.x;
  int cy = blockIdx.y * blockDim.y + threadIdx.y;

  if (cx >= cells_x || cy >= cells_y)
    return;

  int idx = cy * cells_x + cx;

  // Analyze braille cell
  double lumas[8];
  uint8_t colors[8][3];
  bool has_edge;
  double edge_strength;
  double mean_luma;

  analyze_braille_cell_device(
      frame_data, frame_width, frame_height, frame_stride,
      cx, cy, w_scaled, h_scaled, detail_level,
      lumas, colors, has_edge, edge_strength, mean_luma);

  // Determine mode based on edge detection
  if (has_edge)
  {
    // HIGH DETAIL: Use braille
    modes[idx] = 1;

    // Regional color analysis
    uint8_t regional_fg_r, regional_fg_g, regional_fg_b;
    uint8_t regional_bg_r, regional_bg_g, regional_bg_b;

    analyze_regional_colors_device(
        frame_data, frame_width, frame_height, frame_stride,
        cx, cy, w_scaled, h_scaled, cells_x, cells_y, detail_level,
        regional_fg_r, regional_fg_g, regional_fg_b,
        regional_bg_r, regional_bg_g, regional_bg_b);

    // Calculate pattern
    uint8_t pattern = calculate_braille_pattern_device(
        lumas, colors, detail_level, threshold_steps, mean_luma,
        regional_bg_r, regional_bg_g, regional_bg_b);

    // Calculate colors
    uint8_t fg_r, fg_g, fg_b, bg_r, bg_g, bg_b;
    calculate_braille_colors_device(
        colors, lumas, pattern, detail_level,
        fg_r, fg_g, fg_b, bg_r, bg_g, bg_b);

    patterns[idx] = pattern;
    fg_colors[idx * 3 + 0] = fg_r;
    fg_colors[idx * 3 + 1] = fg_g;
    fg_colors[idx * 3 + 2] = fg_b;
    bg_colors[idx * 3 + 0] = bg_r;
    bg_colors[idx * 3 + 1] = bg_g;
    bg_colors[idx * 3 + 2] = bg_b;
  }
  else
  {
    // LOW DETAIL: Use half-block
    modes[idx] = 0;

    // Top half (dots 0,1,3,4)
    uint32_t top_r = colors[0][0] + colors[1][0] + colors[3][0] + colors[4][0];
    uint32_t top_g = colors[0][1] + colors[1][1] + colors[3][1] + colors[4][1];
    uint32_t top_b = colors[0][2] + colors[1][2] + colors[3][2] + colors[4][2];

    // Bottom half (dots 2,5,6,7)
    uint32_t bot_r = colors[2][0] + colors[5][0] + colors[6][0] + colors[7][0];
    uint32_t bot_g = colors[2][1] + colors[5][1] + colors[6][1] + colors[7][1];
    uint32_t bot_b = colors[2][2] + colors[5][2] + colors[6][2] + colors[7][2];

    patterns[idx] = 0; // Not used for half-block
    fg_colors[idx * 3 + 0] = top_r / 4;
    fg_colors[idx * 3 + 1] = top_g / 4;
    fg_colors[idx * 3 + 2] = top_b / 4;
    bg_colors[idx * 3 + 0] = bot_r / 4;
    bg_colors[idx * 3 + 1] = bot_g / 4;
    bg_colors[idx * 3 + 2] = bot_b / 4;
  }
}

// ============================================================================
// HOST FUNCTIONS
// ============================================================================

extern "C"
{
  void cuda_render_braille(
      const uint8_t *d_frame_data,
      uint32_t frame_width,
      uint32_t frame_height,
      uint32_t frame_stride,
      int w_scaled,
      int h_scaled,
      int cells_x,
      int cells_y,
      uint8_t detail_level,
      uint8_t threshold_steps,
      uint8_t *d_patterns,
      uint8_t *d_fg_colors,
      uint8_t *d_bg_colors)
  {
    dim3 block(16, 16);
    dim3 grid((cells_x + block.x - 1) / block.x,
              (cells_y + block.y - 1) / block.y);

    render_braille_kernel<<<grid, block>>>(
        d_frame_data, frame_width, frame_height, frame_stride,
        w_scaled, h_scaled, cells_x, cells_y,
        detail_level, threshold_steps,
        d_patterns, d_fg_colors, d_bg_colors);
  }

  void cuda_render_hybrid(
      const uint8_t *d_frame_data,
      uint32_t frame_width,
      uint32_t frame_height,
      uint32_t frame_stride,
      int w_scaled,
      int h_scaled,
      int cells_x,
      int cells_y,
      uint8_t detail_level,
      uint8_t threshold_steps,
      uint8_t *d_modes,
      uint8_t *d_patterns,
      uint8_t *d_fg_colors,
      uint8_t *d_bg_colors)
  {
    dim3 block(16, 16);
    dim3 grid((cells_x + block.x - 1) / block.x,
              (cells_y + block.y - 1) / block.y);

    render_hybrid_kernel<<<grid, block>>>(
        d_frame_data, frame_width, frame_height, frame_stride,
        w_scaled, h_scaled, cells_x, cells_y,
        detail_level, threshold_steps,
        d_modes, d_patterns, d_fg_colors, d_bg_colors);
  }

} // extern "C"