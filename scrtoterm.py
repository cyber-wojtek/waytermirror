#!/usr/bin/env python3
"""
Wayland Screen to Terminal Display - OPTIMIZED VERSION
Captures Wayland display and renders it in the terminal with various modes.
Supports input forwarding to interact with the captured display.

Optimized for performance with minimal overhead.
"""

import sys
import subprocess
import argparse
import time
import os
import termios
import tty
from PIL import Image
from io import BytesIO
import select

# ANSI color codes
RESET = '\033[0m'

# Global cache for color conversions
_COLOR_CACHE = {}
_COLOR_CACHE_MAX = 4096

# Pre-computed lookup tables
_CUBE_LEVELS = [0, 95, 135, 175, 215, 255]
_ASCII_CHARS = ' .\'`^",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$'
_BRAILLE_BASE = 0x2800
_DOT_MAP = [0x01, 0x08, 0x02, 0x10, 0x04, 0x20, 0x40, 0x80]

def _rgb_to_ansi_256_impl(r, g, b):
    """Internal implementation of RGB to 256 color conversion"""
    # Find nearest cube level
    ri = min(range(6), key=lambda i: abs(_CUBE_LEVELS[i] - r))
    gi = min(range(6), key=lambda i: abs(_CUBE_LEVELS[i] - g))
    bi = min(range(6), key=lambda i: abs(_CUBE_LEVELS[i] - b))
    
    cr, cg, cb = _CUBE_LEVELS[ri], _CUBE_LEVELS[gi], _CUBE_LEVELS[bi]
    cube_index = 16 + 36 * ri + 6 * gi + bi
    cube_dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
    
    # Grayscale candidate
    avg = (r + g + b) // 3
    if avg < 8:
        gray_index, gr, gg, gb = 16, 0, 0, 0
    elif avg > 238:
        gray_index, gr, gg, gb = 231, 255, 255, 255
    else:
        gi = max(0, min(23, round((avg - 8) / 10)))
        gray_index = 232 + gi
        gr = gg = gb = 8 + gi * 10
    
    gray_dist = (r - gr) ** 2 + (g - gg) ** 2 + (b - gb) ** 2
    return gray_index if gray_dist < cube_dist else cube_index

def rgb_to_ansi_256(r, g, b):
    """Convert RGB to ANSI 256 color with caching"""
    # Round to nearest lookup value for cache hits
    r_round = (r // 17) * 17
    g_round = (g // 17) * 17
    b_round = (b // 17) * 17
    key = (r_round, g_round, b_round)
    
    if key in _RGB256_LOOKUP:
        return _RGB256_LOOKUP[key]
    
    # Fallback to full calculation
    return _rgb_to_ansi_256_impl(r, g, b)

def rgb_to_ansi_16(r, g, b):
    """Convert RGB to ANSI 16-color palette (cached)"""
    cache_key = (r >> 4, g >> 4, b >> 4, 16)  # Reduce precision for better cache hits
    
    if cache_key in _COLOR_CACHE:
        return _COLOR_CACHE[cache_key]
    
    luminance = int(0.299 * r + 0.587 * g + 0.114 * b)
    
    # Grayscale
    if abs(r - g) < 30 and abs(g - b) < 30:
        if luminance < 32:
            result = 30
        elif luminance < 96:
            result = 90
        elif luminance < 192:
            result = 37
        else:
            result = 97
    else:
        bright = luminance > 128
        max_val = max(r, g, b)
        if max_val == 0:
            result = 30
        else:
            r_ratio = r / max_val
            g_ratio = g / max_val
            b_ratio = b / max_val
            threshold = 0.6
            
            # Determine color
            if r_ratio > threshold and g_ratio > threshold and b_ratio < threshold:
                color = 3  # Yellow
            elif r_ratio > threshold and b_ratio > threshold and g_ratio < threshold:
                color = 5  # Magenta
            elif g_ratio > threshold and b_ratio > threshold and r_ratio < threshold:
                color = 6  # Cyan
            elif r_ratio > threshold and g_ratio < threshold and b_ratio < threshold:
                color = 1  # Red
            elif g_ratio > threshold and r_ratio < threshold and b_ratio < threshold:
                color = 2  # Green
            elif b_ratio > threshold and r_ratio < threshold and g_ratio < threshold:
                color = 4  # Blue
            else:
                result = 30 if luminance < 128 else 37
                color = None
            
            if color is not None:
                result = (90 if bright else 30) + color
    
    # Cache result
    if len(_COLOR_CACHE) < _COLOR_CACHE_MAX:
        _COLOR_CACHE[cache_key] = result
    
    return result

def get_ansi_color(r, g, b, color_mode):
    """Get ANSI color code based on color mode (optimized with caching)"""
    if color_mode == 'truecolor':
        return f'\033[38;2;{r};{g};{b}m'
    elif color_mode == '256':
        code = rgb_to_ansi_256(r, g, b)
        return f'\033[38;5;{code}m'
    else:  # 16 color
        code = rgb_to_ansi_16(r, g, b)
        return f'\033[{code}m'

def get_wayland_displays():
    """List available Wayland displays (cached result)"""
    xdg_runtime = os.environ.get('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')
    
    if not os.path.exists(xdg_runtime):
        return []
    
    return sorted([item for item in os.listdir(xdg_runtime) if item.startswith('wayland-')])


class WaylandCaptureSession:
    """Optimized Wayland capture using grim (fast) or wf-recorder fallback"""
    
    def __init__(self, display=None):
        self.display = display or self._auto_detect_display()
        self.env = self._setup_environment()
        self.last_frame = None
        self.frame_count = 0
        self.use_grim = self._check_grim()
        
        if not self.display:
            raise RuntimeError("No Wayland display found")
    
    def _auto_detect_display(self):
        """Auto-detect Wayland display"""
        display = os.environ.get('WAYLAND_DISPLAY')
        if display:
            return display
        
        xdg_runtime = os.environ.get('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')
        if os.path.exists(xdg_runtime):
            for item in os.listdir(xdg_runtime):
                if item.startswith("wayland-"):
                    return item
        return None
    
    def _setup_environment(self):
        """Setup environment variables"""
        env = os.environ.copy()
        env['WAYLAND_DISPLAY'] = self.display or 'wayland-0'
        env['XDG_RUNTIME_DIR'] = f"/run/user/{os.getuid()}"
        return env
    
    def _check_grim(self):
        """Check if grim is available"""
        try:
            subprocess.run(['grim'], capture_output=True, timeout=1, check=True)
            return True
        except:
            return False
    
    def start(self):
        """Initialize session"""
        method = "grim" if self.use_grim else "wf-recorder"
        print(f"Capture session using {method} for {self.display}", file=sys.stderr)
    
    def capture_frame(self):
        """Capture a single frame"""
        if self.use_grim:
            return self._capture_grim()
        else:
            return self._capture_wfrecorder()
    
    def _capture_grim(self):
        """Fast capture using grim"""
        try:
            result = subprocess.run(
                ['grim', '-t', 'png', '-l', '0', '-'],
                env=self.env,
                capture_output=True,
                timeout=1.0,
                check=False
            )
            
            if result.returncode == 0 and result.stdout:
                self.last_frame = Image.open(BytesIO(result.stdout))
                self.frame_count += 1
                return self.last_frame
        except:
            pass
        
        return self.last_frame
    
    def _capture_wfrecorder(self):
        """Fallback capture using wf-recorder"""
        try:
            recorder = subprocess.Popen(
                ['wf-recorder', '-c', 'rawvideo', '-m', 'avi', '-t', '0.05', '-f', '-'],
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0
            )
            
            ffmpeg = subprocess.Popen(
                ['ffmpeg', '-f', 'avi', '-i', 'pipe:0', '-frames:v', '1', 
                 '-f', 'image2pipe', '-vcodec', 'png', '-'],
                stdin=recorder.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            
            recorder.stdout.close()
            
            try:
                png_data, _ = ffmpeg.communicate(timeout=1.5)
            except subprocess.TimeoutExpired:
                ffmpeg.kill()
                recorder.kill()
                return self.last_frame
            
            recorder.terminate()
            try:
                recorder.wait(timeout=0.3)
            except subprocess.TimeoutExpired:
                recorder.kill()
            
            if png_data:
                self.last_frame = Image.open(BytesIO(png_data))
                self.frame_count += 1
        except:
            pass
        
        return self.last_frame
    
    def stop(self):
        """Stop the capture session"""
        pass
    
    def __del__(self):
        self.stop()


def capture_wayland_screen_fast(display=None):
    """Fast one-shot capture"""
    env = os.environ.copy()
    
    if not display:
        display = os.environ.get('WAYLAND_DISPLAY')
        if not display:
            xdg_runtime = env.get('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')
            for item in os.listdir(xdg_runtime):
                if item.startswith("wayland-"):
                    display = item
                    break
    
    if not display:
        return None

    env['WAYLAND_DISPLAY'] = display
    env['XDG_RUNTIME_DIR'] = f"/run/user/{os.getuid()}"
    
    # Try grim first
    try:
        result = subprocess.run(
            ['grim', '-t', 'png', '-'],
            env=env,
            capture_output=True,
            timeout=1.0,
            check=False
        )
        
        if result.returncode == 0 and result.stdout:
            return Image.open(BytesIO(result.stdout))
    except:
        pass
    
    # Fallback to wf-recorder
    try:
        recorder = subprocess.Popen(
            ['wf-recorder', '-c', 'rawvideo', '-m', 'avi', '-t', '0.1', '-f', '-'],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        
        ffmpeg = subprocess.Popen(
            ['ffmpeg', '-f', 'avi', '-i', '-', '-frames:v', '1', 
             '-f', 'image2pipe', '-vcodec', 'png', '-'],
            stdin=recorder.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        
        recorder.stdout.close()
        png_data, _ = ffmpeg.communicate(timeout=1.5)
        
        recorder.terminate()
        try:
            recorder.wait(timeout=0.3)
        except subprocess.TimeoutExpired:
            recorder.kill()
        
        if png_data:
            return Image.open(BytesIO(png_data))
    except:
        pass
    
    return None

def get_display_geometry(display=None):
    """Get display geometry (cached where possible)"""
    env = os.environ.copy()
    if display:
        env['WAYLAND_DISPLAY'] = display
    
    try:
        result = subprocess.run(
            ['wlr-randr'],
            capture_output=True,
            text=True,
            env=env,
            timeout=2,
            check=False
        )
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'current' in line.lower():
                    parts = line.strip().split()
                    if parts and 'x' in parts[0]:
                        res_part = parts[0].replace('px,', '').replace('px', '')
                        if 'x' in res_part:
                            try:
                                width, height = map(int, res_part.split('x'))
                                return width, height
                            except ValueError:
                                pass
    except:
        pass
    
    # Fallback: capture and detect
    try:
        img = capture_wayland_screen_fast(display)
        if img:
            return img.width, img.height
    except:
        pass
    
    return 1920, 1080

def send_mouse_click(x, y, button=1, display=None):
    """Send mouse click (optimized with reduced error checking)"""
    env = os.environ.copy()
    if display:
        env['WAYLAND_DISPLAY'] = display
    
    try:
        subprocess.run(['ydotool', 'mousemove', '--absolute', str(x), str(y)], 
                      env=env, check=False, stderr=subprocess.DEVNULL)
        subprocess.run(['ydotool', 'click', f'0x{button:02x}'], 
                      env=env, check=False, stderr=subprocess.DEVNULL)
    except:
        pass

def send_key(key, display=None):
    """Send key press (optimized with static key map)"""
    env = os.environ.copy()
    if display:
        env['WAYLAND_DISPLAY'] = display
    
    _KEY_MAP = {
        '\r': '0x1c', '\n': '0x1c', '\t': '0x0f', '\x1b': '0x01', ' ': '0x39',
        'a': '0x1e', 'b': '0x30', 'c': '0x2e', 'd': '0x20', 'e': '0x12',
        'f': '0x21', 'g': '0x22', 'h': '0x23', 'i': '0x17', 'j': '0x24',
        'k': '0x25', 'l': '0x26', 'm': '0x32', 'n': '0x31', 'o': '0x18',
        'p': '0x19', 'q': '0x10', 'r': '0x13', 's': '0x1f', 't': '0x14',
        'u': '0x16', 'v': '0x2f', 'w': '0x11', 'x': '0x2d', 'y': '0x15',
        'z': '0x2c',
    }
    
    try:
        key_lower = key.lower()
        if key_lower in _KEY_MAP:
            subprocess.run(['ydotool', 'key', _KEY_MAP[key_lower]], 
                          env=env, check=False, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(['ydotool', 'type', key], 
                          env=env, check=False, stderr=subprocess.DEVNULL)
    except:
        pass

def parse_mouse_input(seq):
    """Parse xterm mouse reporting sequences"""
    if len(seq) < 6:
        return None
    
    button = ord(seq[3]) - 32
    x = ord(seq[4]) - 33
    y = ord(seq[5]) - 33
    
    return (button, x, y)

# Keep all rendering functions exactly as they were
def render_braille(image, width, height, color_mode):
    """Render image using Unicode braille characters"""
    braille_base = 0x2800
    dot_map = [0x01, 0x08, 0x02, 0x10, 0x04, 0x20, 0x40, 0x80]
    
    img_width = width * 2
    img_height = height * 4
    img = image.convert('RGB')
    img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
    pixels = img.load()
    
    luma_map = {}
    for py in range(img_height):
        for px in range(img_width):
            r, g, b = pixels[px, py][:3]
            luma = 0.299 * r + 0.587 * g + 0.114 * b
            luma_map[(px, py)] = luma
    
    output = []
    for y in range(height):
        line = []
        for x in range(width):
            pattern = 0
            colors = []
            lumas = []
            
            for dy in range(4):
                for dx in range(2):
                    px = x * 2 + dx
                    py = y * 4 + dy
                    if px < img_width and py < img_height:
                        r, g, b = pixels[px, py][:3]
                        colors.append((r, g, b))
                        lumas.append(luma_map[(px, py)])
            
            if not lumas:
                continue
            
            avg_luma = sum(lumas) / len(lumas)
            variance = sum((l - avg_luma) ** 2 for l in lumas) / len(lumas)
            edge_strength = min(variance / 1000.0, 1.0)
            
            sorted_lumas = sorted(lumas)
            median_luma = sorted_lumas[len(sorted_lumas) // 2]
            threshold = avg_luma * (1 - edge_strength * 0.5) + median_luma * (edge_strength * 0.5)
            
            idx = 0
            for dy in range(4):
                for dx in range(2):
                    px = x * 2 + dx
                    py = y * 4 + dy
                    if px < img_width and py < img_height:
                        luma = luma_map[(px, py)]
                        if variance > 1000:
                            if luma > threshold * 1.08:
                                pattern |= dot_map[idx]
                        else:
                            if luma > threshold * 0.95:
                                pattern |= dot_map[idx]
                    idx += 1
            
            avg_r = sum(c[0] for c in colors) // len(colors)
            avg_g = sum(c[1] for c in colors) // len(colors)
            avg_b = sum(c[2] for c in colors) // len(colors)
            
            boost_factor = 2.5
            avg_r = min(255, int(avg_r * boost_factor))
            avg_g = min(255, int(avg_g * boost_factor))
            avg_b = min(255, int(avg_b * boost_factor))
            
            color = get_ansi_color(avg_r, avg_g, avg_b, color_mode)
            braille_char = chr(braille_base + pattern)
            line.append(f"{color}{braille_char}")
        
        output.append(''.join(line) + RESET)
    
    return '\n'.join(output)

def render_blocks(image, width, height, color_mode):
    """Render image using ASCII block characters (half blocks)"""
    img_width = width
    img_height = height * 2
    img = image.resize((img_width, img_height), Image.Resampling.LANCZOS)
    pixels = img.load()
    
    output = []
    for y in range(height):
        line = []
        for x in range(width):
            top_r, top_g, top_b = pixels[x, y * 2][:3]
            
            if y * 2 + 1 < img_height:
                bot_r, bot_g, bot_b = pixels[x, y * 2 + 1][:3]
            else:
                bot_r, bot_g, bot_b = top_r, top_g, top_b
            
            top_color = get_ansi_color(top_r, top_g, top_b, color_mode)
            
            if color_mode == 'truecolor':
                bot_bg = f'\033[48;2;{bot_r};{bot_g};{bot_b}m'
            elif color_mode == '256':
                bot_code = rgb_to_ansi_256(bot_r, bot_g, bot_b)
                bot_bg = f'\033[48;5;{bot_code}m'
            else:
                bot_code = rgb_to_ansi_16(bot_r, bot_g, bot_b)
                bot_bg = f'\033[{bot_code + 10}m'
            
            line.append(f"{bot_bg}{top_color}▀")
        
        output.append(''.join(line) + RESET)
    
    return '\n'.join(output)

def render_ascii(image, width, height, color_mode):
    """Render image using ASCII characters"""
    ascii_chars = ' .\'`^",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$'
    
    img = image.convert('RGB')
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    pixels = img.load()
    
    all_lumas = []
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y][:3]
            luma = 0.299 * r + 0.587 * g + 0.114 * b
            all_lumas.append(luma)
    
    min_luma = min(all_lumas)
    max_luma = max(all_lumas)
    luma_range = max_luma - min_luma
    
    output = []
    for y in range(height):
        line = []
        for x in range(width):
            r, g, b = pixels[x, y][:3]
            luma = 0.299 * r + 0.587 * g + 0.114 * b
            
            if luma_range > 0:
                normalized = (luma - min_luma) / luma_range
            else:
                normalized = 0.5
            
            normalized = normalized * normalized * (3 - 2 * normalized)
            
            char_index = int(normalized * (len(ascii_chars) - 1))
            char_index = max(0, min(len(ascii_chars) - 1, char_index))
            char = ascii_chars[char_index]
            
            boost = 1.8
            r_boost = min(255, int(r * boost))
            g_boost = min(255, int(g * boost))
            b_boost = min(255, int(b * boost))
            
            color = get_ansi_color(r_boost, g_boost, b_boost, color_mode)
            line.append(f"{color}{char}")
        
        output.append(''.join(line) + RESET)
    
    return '\n'.join(output)

def render_mixed(image, width, height, color_mode):
    """Mixed mode: intelligently chooses the best glyph per cell"""
    ascii_chars = ' .\'`^",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$'

    img = image.convert('RGB')
    img = img.resize((width * 2, height * 4), Image.Resampling.LANCZOS)
    pixels = img.load()

    braille_base = 0x2800
    dot_map = [0x01, 0x08, 0x02, 0x10, 0x04, 0x20, 0x40, 0x80]

    output = []

    for y in range(height):
        line = []
        for x in range(width):
            colors = []
            lumas = []
            for dy in range(4):
                for dx in range(2):
                    px = x * 2 + dx
                    py = y * 4 + dy
                    r, g, b = pixels[px, py][:3]
                    colors.append((r, g, b))
                    luma = 0.299*r + 0.587*g + 0.114*b
                    lumas.append(luma)

            avg_luma = sum(lumas) / len(lumas)
            variance = sum((l - avg_luma)**2 for l in lumas) / len(lumas)
            
            avg_r = sum(c[0] for c in colors) // len(colors)
            avg_g = sum(c[1] for c in colors) // len(colors)
            avg_b = sum(c[2] for c in colors) // len(colors)
            
            brightness = avg_luma / 255.0
            detail = min(variance / 2500.0, 1.0)
            
            score_block = 0.75 * brightness + 0.25 * (1 - detail)
            score_ascii = 0.6 * (1 - abs(brightness - 0.5)) + 0.4 * detail
            score_braille = 0.65 * detail + 0.35 * (1 - brightness)
            
            if 0.3 < brightness < 0.7:
                score_ascii *= 1.3
            if detail > 0.5:
                score_braille *= 1.2

            mode = max([
                ("block", score_block),
                ("ascii", score_ascii),
                ("braille", score_braille)
            ], key=lambda x: x[1])[0]

            if mode == "block":
                char = "█"
                color = get_ansi_color(avg_r, avg_g, avg_b, color_mode)
                
            elif mode == "ascii":
                char_idx = int(brightness * (len(ascii_chars) - 1))
                char = ascii_chars[char_idx]
                boost = 2.0
                color = get_ansi_color(
                    min(255, int(avg_r * boost)),
                    min(255, int(avg_g * boost)),
                    min(255, int(avg_b * boost)),
                    color_mode
                )
                
            else:  # braille
                pattern = 0
                sorted_lumas = sorted(lumas)
                threshold = sorted_lumas[len(sorted_lumas) // 2]
                
                for idx, luma in enumerate(lumas):
                    if luma > threshold * 1.05:
                        pattern |= dot_map[idx]
                
                char = chr(braille_base + pattern)
                boost = 2.5
                color = get_ansi_color(
                    min(255, int(avg_r * boost)),
                    min(255, int(avg_g * boost)),
                    min(255, int(avg_b * boost)),
                    color_mode
                )

            line.append(f"{color}{char}")
        output.append(''.join(line) + RESET)

    return '\n'.join(output)

def get_terminal_size():
    """Get terminal size in characters"""
    try:
        result = subprocess.run(['stty', 'size'], capture_output=True, text=True, check=False)
        rows, cols = map(int, result.stdout.split())
        return cols, rows
    except:
        return 80, 24

def clear_screen():
    """Clear terminal screen"""
    print('\033[2J\033[H', end='')

def enable_mouse_tracking():
    """Enable xterm mouse tracking"""
    sys.stdout.write('\033[?1000h\033[?1002h\033[?1015h\033[?1006h')
    sys.stdout.flush()

def disable_mouse_tracking():
    """Disable xterm mouse tracking"""
    sys.stdout.write('\033[?1006l\033[?1015l\033[?1002l\033[?1000l')
    sys.stdout.flush()

def main():
    # Build lookup tables at startup
    _build_rgb256_lookup()
    
    parser = argparse.ArgumentParser(description='Display Wayland screen in terminal (OPTIMIZED)')
    parser.add_argument('-m', '--mode', choices=['braille', 'blocks', 'ascii', 'mixed'],
                    default='blocks', help='Character mode (default: blocks)')
    parser.add_argument('-c', '--color', choices=['truecolor', '256', '16'],
                       default='truecolor', help='Color mode (default: truecolor)')
    parser.add_argument('-f', '--fps', type=float, default=10.0,
                       help='Frames per second (default: 10.0)')
    parser.add_argument('-w', '--width', type=int, help='Terminal width (auto-detect if not set)')
    parser.add_argument('-H', '--height', type=int, help='Terminal height (auto-detect if not set)')
    parser.add_argument('--once', action='store_true', help='Capture once and exit')
    parser.add_argument('-d', '--display', type=str, help='Wayland display (e.g., wayland-0)')
    parser.add_argument('--list-displays', action='store_true', help='List available displays')
    parser.add_argument('-i', '--input', action='store_true', help='Enable input forwarding')
    parser.add_argument('--session', action='store_true', 
                       help='Use capture session (recommended)')
    
    args = parser.parse_args()
    
    if args.list_displays:
        displays = get_wayland_displays()
        if displays:
            print("Available Wayland displays:")
            for disp in displays:
                print(f"  {disp}")
        else:
            print("No Wayland displays found")
        return
    
    # Get terminal size
    if args.width and args.height:
        term_width, term_height = args.width, args.height
    else:
        term_width, term_height = get_terminal_size()
        term_width -= 2
        term_height -= 3
    
    display_str = args.display if args.display else os.environ.get('WAYLAND_DISPLAY', 'default')
    print(f"Wayland display: {display_str}")
    print(f"Terminal size: {term_width}x{term_height}")
    print(f"Mode: {args.mode}, Colors: {args.color}, FPS: {args.fps}")
    if args.session:
        print("Using capture session mode")
    if args.input:
        print("Input forwarding: ENABLED (press 'q' to quit)")
    else:
        print("Press Ctrl+C to exit")
    print()
    time.sleep(1)
    
    # Setup terminal for input if enabled
    old_settings = None
    if args.input:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        enable_mouse_tracking()
    
    # Create capture session if requested
    capture_session = None
    if args.session:
        try:
            capture_session = WaylandCaptureSession(args.display)
            capture_session.start()
        except Exception as e:
            print(f"Failed to create capture session: {e}", file=sys.stderr)
            print("Falling back to one-shot mode", file=sys.stderr)
            capture_session = None
    
    try:
        display_width, display_height = get_display_geometry(args.display)
        print(f"Display resolution: {display_width}x{display_height}")
        
        frame_times = []
        
        while True:
            start_time = time.time()
            
            # Check for input if enabled
            if args.input and select.select([sys.stdin], [], [], 0)[0]:
                char = sys.stdin.read(1)
                
                if char == 'q':
                    break
                
                if char == '\x1b':
                    if select.select([sys.stdin], [], [], 0.01)[0]:
                        next_char = sys.stdin.read(1)
                        if next_char == '[':
                            seq = '\x1b['
                            while True:
                                if select.select([sys.stdin], [], [], 0.01)[0]:
                                    c = sys.stdin.read(1)
                                    seq += c
                                    if c in 'ABCDEFGHM~':
                                        break
                                else:
                                    break
                            
                            if 'M' in seq:
                                mouse_data = parse_mouse_input(seq)
                                if mouse_data:
                                    button, x, y = mouse_data
                                    real_x = int(x * display_width / term_width)
                                    real_y = int(y * display_height / term_height)
                                    send_mouse_click(real_x, real_y, button, args.display)
                        else:
                            send_key('\x1b', args.display)
                else:
                    send_key(char, args.display)
            
            # Capture screen
            if capture_session:
                image = capture_session.capture_frame()
            else:
                image = capture_wayland_screen_fast(args.display)
            
            if image is None:
                if capture_session:
                    time.sleep(0.1)
                    continue
                else:
                    print("Failed to capture frame", file=sys.stderr)
                    break
            
            # Render based on mode   
            if args.mode == 'braille':
                output = render_braille(image, term_width, term_height, args.color)
            elif args.mode == 'blocks':
                output = render_blocks(image, term_width, term_height, args.color)
            elif args.mode == 'ascii':
                output = render_ascii(image, term_width, term_height, args.color)
            else:  # mixed
                output = render_mixed(image, term_width, term_height, args.color)
                
            # Display
            clear_screen()
            print(output)
            
            # Calculate FPS
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            if len(frame_times) > 10:
                frame_times.pop(0)
            avg_frame_time = sum(frame_times) / len(frame_times)
            actual_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            status = f"\nMode: {args.mode} | Colors: {args.color} | Display: {display_str}"
            status += f" | FPS: {actual_fps:.1f} ({frame_time*1000:.0f}ms/frame)"
            if args.input:
                status += " | Input: ON (q=quit)"
            print(status)
            
            if args.once:
                break
            
            # Wait for next frame
            sleep_time = max(0, (1.0 / args.fps) - frame_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        # Cleanup
        if capture_session:
            capture_session.stop()
        if args.input and old_settings:
            disable_mouse_tracking()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print(RESET)

if __name__ == '__main__':
    main()
