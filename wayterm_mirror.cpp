#include <wayland-client.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cstring>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <sys/ioctl.h>
#include <termios.h>
#include <linux/input-event-codes.h>
#include <argparse/argparse.hpp>
#include "wlr-screencopy-unstable-v1-client-protocol.h"
#include "virtual-keyboard-unstable-v1-client-protocol.h"
#include "wlr-virtual-pointer-unstable-v1-client-protocol.h"

// --- Wayland globals ---
static wl_display * display;
static wl_registry * registry;
static wl_shm * shm;
static wl_output * output;
static wl_seat * seat;
static zwlr_screencopy_manager_v1 * manager;
static zwlr_virtual_pointer_manager_v1 * pointer_manager;
static zwp_virtual_keyboard_manager_v1 * keyboard_manager;
static zwlr_virtual_pointer_v1 * virtual_pointer;
static zwp_virtual_keyboard_v1 * virtual_keyboard;
static std::vector<wl_output*> outputs;
static int selected_output_index = 0;
static bool feature_video = true;
static bool feature_audio = true;
static bool feature_input = true;

struct Capture {
  wl_buffer * buffer;
  void * data;
  int width, height;
  int stride;
  int size;
  uint32_t format;
};
static Capture capture;
static bool done = false;

// --- Options ---
enum class ColorMode {
  ANSI_16,
  ANSI_256,
  ANSI_TRUECOLOR
};
ColorMode mode = ColorMode::ANSI_256;
std::string renderer = "braille";
double scale_factor = 1.0;
bool auto_fit = false;
bool input_enabled = false;

// --- Terminal state ---
static struct termios orig_termios;
static bool terminal_raw_mode = false;

// --- Mouse state ---
static int mouse_x = 0, mouse_y = 0;
static int term_width = 80, term_height = 24;

// --- Helpers ---
static int create_shm_file(size_t size) {
  int fd = memfd_create("wayterm-shm", MFD_CLOEXEC);
  if (fd < 0) return -1;
  if (ftruncate(fd, size) < 0) {
    close(fd);
    return -1;
  }
  return fd;
}

// --- Terminal raw mode ---
static void enable_raw_mode() {
  if (terminal_raw_mode) return;
  tcgetattr(STDIN_FILENO, &orig_termios);
  struct termios raw = orig_termios;
  
  // Disable canonical mode and echo
  raw.c_lflag &= ~(ECHO | ICANON | ISIG);
  
  // Don't disable IEXTEN - needed for some keys
  // Keep input processing minimal
  raw.c_iflag &= ~(IXON | ICRNL | BRKINT | INPCK | ISTRIP);
  
  // Keep output processing - important!
  // Don't touch c_oflag
  
  // Set read to return immediately
  raw.c_cc[VMIN] = 0;
  raw.c_cc[VTIME] = 1; // 100ms timeout
  
  tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
  terminal_raw_mode = true;
  
  // Enable mouse tracking
  std::cout << "\033[?1000h\033[?1006h" << std::flush;
}

static void disable_raw_mode() {
  if (!terminal_raw_mode) return;
  // Disable mouse tracking
  std::cout << "\033[?1000l\033[?1006l" << std::flush;
  tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
  terminal_raw_mode = false;
}

// --- RGB -> ANSI ---
static uint8_t rgb_to_ansi_16(uint8_t r, uint8_t g, uint8_t b) {
  uint8_t intensity = (r > 128 || g > 128 || b > 128) ? 8 : 0;
  uint8_t code = intensity;
  if (r > 64) code |= 1;
  if (g > 64) code |= 2;
  if (b > 64) code |= 4;
  return code;
}

static uint8_t rgb_to_ansi_256(uint8_t r, uint8_t g, uint8_t b) {
  int max_rgb = std::max({r, g, b});
  int min_rgb = std::min({r, g, b});
  int saturation = max_rgb - min_rgb;
  
  if (saturation < 30) {
    int avg = (r + g + b) / 3;
    if (avg < 8) return 16;
    if (avg > 238) return 231;
    return 232 + (avg - 8) * 24 / 230;
  }
  
  const int CUBE[6] = {0, 95, 135, 175, 215, 255};
  int ri = 0, gi = 0, bi = 0;
  int min_r = 256, min_g = 256, min_b = 256;
  for (int i = 0; i < 6; i++) {
    int dr = abs(CUBE[i] - r);
    if (dr < min_r) { min_r = dr; ri = i; }
    int dg = abs(CUBE[i] - g);
    if (dg < min_g) { min_g = dg; gi = i; }
    int db = abs(CUBE[i] - b);
    if (db < min_b) { min_b = db; bi = i; }
  }
  
  return 16 + 36 * ri + 6 * gi + bi;
}

static std::string rgb_to_ansi(uint8_t r, uint8_t g, uint8_t b, ColorMode m) {
  switch (m) {
  case ColorMode::ANSI_16:
    return "\033[38;5;" + std::to_string(rgb_to_ansi_16(r, g, b)) + "m";
  case ColorMode::ANSI_256:
    return "\033[38;5;" + std::to_string(rgb_to_ansi_256(r, g, b)) + "m";
  case ColorMode::ANSI_TRUECOLOR:
    return "\033[38;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) + "m";
  default:
    return "";
  }
}

// --- Frame listeners ---
static void frame_buffer(void * , zwlr_screencopy_frame_v1 * frame,
  uint32_t format, uint32_t width, uint32_t height, uint32_t stride) {
  capture.width = width;
  capture.height = height;
  capture.stride = stride;
  capture.size = stride * height;
  capture.format = format;
  int fd = create_shm_file(capture.size);
  if (fd < 0) return;
  capture.data = mmap(nullptr, capture.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (capture.data == MAP_FAILED) {
    close(fd);
    return;
  }
  wl_shm_pool * pool = wl_shm_create_pool(shm, fd, capture.size);
  capture.buffer = wl_shm_pool_create_buffer(pool, 0, width, height, stride, format);
  wl_shm_pool_destroy(pool);
  close(fd);
  zwlr_screencopy_frame_v1_copy(frame, capture.buffer);
}

static void frame_flags(void * , zwlr_screencopy_frame_v1 * , uint32_t) {}

static void frame_ready(void * , zwlr_screencopy_frame_v1 * frame, uint32_t, uint32_t, uint32_t) {
  done = true;
  zwlr_screencopy_frame_v1_destroy(frame);
}

static void frame_failed(void *, zwlr_screencopy_frame_v1 * frame) {
  done = true;
  zwlr_screencopy_frame_v1_destroy(frame);
}

static const zwlr_screencopy_frame_v1_listener frame_listener = {
  frame_buffer,
  frame_flags,
  frame_ready,
  frame_failed
};

// --- Registry ---
static void registry_add(void * , wl_registry * r, uint32_t name,
  const char * iface, uint32_t version) {
  if (strcmp(iface, wl_shm_interface.name) == 0) 
    shm = static_cast<wl_shm*>(wl_registry_bind(r, name, &wl_shm_interface, 1));
  else if (strcmp(iface, wl_output_interface.name) == 0) {
    wl_output* out = static_cast<wl_output*>(wl_registry_bind(r, name, &wl_output_interface, 1));
    outputs.push_back(out);
  }
  else if (strcmp(iface, wl_seat_interface.name) == 0)
    seat = static_cast<wl_seat*>(wl_registry_bind(r, name, &wl_seat_interface, 1));
  else if (strcmp(iface, zwlr_screencopy_manager_v1_interface.name) == 0)
    manager = static_cast<zwlr_screencopy_manager_v1*>(wl_registry_bind(r, name, &zwlr_screencopy_manager_v1_interface, 1));
  else if (strcmp(iface, zwlr_virtual_pointer_manager_v1_interface.name) == 0)
    pointer_manager = static_cast<zwlr_virtual_pointer_manager_v1*>(wl_registry_bind(r, name, &zwlr_virtual_pointer_manager_v1_interface, std::min(version, 2u)));
  else if (strcmp(iface, zwp_virtual_keyboard_manager_v1_interface.name) == 0)
    keyboard_manager = static_cast<zwp_virtual_keyboard_manager_v1*>(wl_registry_bind(r, name, &zwp_virtual_keyboard_manager_v1_interface, 1));
}

static void registry_remove(void * , wl_registry * , uint32_t) {}

static const wl_registry_listener registry_listener = {
  registry_add,
  registry_remove
};

// --- Terminal size ---
static void get_terminal_size(int & tw, int & th) {
  struct winsize w {};
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, & w) == 0) {
    tw = w.ws_col;
    th = w.ws_row;
  } else {
    tw = 80;
    th = 24;
  }
}

// Add a flag to track when we're sending input
static bool sending_input = false;


// Improved keyboard handling with proper timing
static void send_key(uint32_t keycode, bool shift = false, bool ctrl = false) {
  if (!virtual_keyboard) return;
  
  std::cerr << "Sending keycode: " << keycode << " shift=" << shift << " ctrl=" << ctrl << std::endl;
  
  sending_input = true;
  
  uint32_t time_ms = 0;
  
  // Press modifiers first with individual flushes
  if (ctrl) {
    zwp_virtual_keyboard_v1_key(virtual_keyboard, time_ms, KEY_LEFTCTRL, 
      WL_KEYBOARD_KEY_STATE_PRESSED);
    wl_display_flush(display);
    usleep(5000); // 5ms delay
  }
  
  if (shift) {
    zwp_virtual_keyboard_v1_key(virtual_keyboard, time_ms, KEY_LEFTSHIFT, 
      WL_KEYBOARD_KEY_STATE_PRESSED);
    wl_display_flush(display);
    usleep(5000); // 5ms delay
  }
  
  // Press main key
  zwp_virtual_keyboard_v1_key(virtual_keyboard, time_ms, keycode, 
    WL_KEYBOARD_KEY_STATE_PRESSED);
  wl_display_flush(display);
  usleep(10000); // 10ms
  
  // Release main key
  zwp_virtual_keyboard_v1_key(virtual_keyboard, time_ms, keycode, 
    WL_KEYBOARD_KEY_STATE_RELEASED);
  wl_display_flush(display);
  usleep(5000); // 5ms
  
  // Release modifiers
  if (shift) {
    zwp_virtual_keyboard_v1_key(virtual_keyboard, time_ms, KEY_LEFTSHIFT, 
      WL_KEYBOARD_KEY_STATE_RELEASED);
    wl_display_flush(display);
    usleep(5000);
  }
  
  if (ctrl) {
    zwp_virtual_keyboard_v1_key(virtual_keyboard, time_ms, KEY_LEFTCTRL, 
      WL_KEYBOARD_KEY_STATE_RELEASED);
    wl_display_flush(display);
    usleep(5000);
  }
  
  sending_input = false;
}

static void terminal_to_screen_coords(int tx, int ty, int& sx, int& sy) {
  int tw, th;
  get_terminal_size(tw, th);
  
  // Calculate scaled dimensions (matching render logic)
  int w_scaled, h_scaled;
  
  if (renderer == "braille" || renderer == "hybrid") {
    // Braille uses 2x4 pixel cells per character
    if (auto_fit) {
      double aspect = (double)capture.width / capture.height;
      double term_aspect = (double)(tw * 2) / (th * 4);
      if (aspect > term_aspect) {
        w_scaled = tw * 2;
        h_scaled = (int)(w_scaled / aspect);
      } else {
        h_scaled = th * 4;
        w_scaled = (int)(h_scaled * aspect);
      }
    } else {
      w_scaled = (int)(capture.width * scale_factor);
      h_scaled = (int)(capture.height * scale_factor);
    }
    w_scaled = std::min(w_scaled, tw * 2);
    h_scaled = std::min(h_scaled, th * 4);
    
    // Rendering fills from top-left, check bounds
    int render_cols = w_scaled / 2;
    int render_rows = h_scaled / 4;
    
    if (tx >= render_cols || ty >= render_rows) {
      sx = sy = -1; // Outside rendered area
      return;
    }
    
    // Direct mapping - render starts at (0,0)
    sx = (tx * 2 * capture.width) / w_scaled;
    sy = (ty * 4 * capture.height) / h_scaled;
    
  } else if (renderer == "blocks") {
    if (auto_fit) {
      double aspect = (double)capture.width / capture.height;
      double term_aspect = (double)tw / (th * 2);
      if (aspect > term_aspect) {
        w_scaled = tw;
        h_scaled = (int)(tw / aspect);
      } else {
        h_scaled = th * 2;
        w_scaled = (int)(h_scaled * aspect);
      }
    } else {
      w_scaled = (int)(capture.width * scale_factor);
      h_scaled = (int)(capture.height * scale_factor);
    }
    w_scaled = std::min(w_scaled, tw);
    h_scaled = std::min(h_scaled, th * 2);
    
    int render_cols = w_scaled;
    int render_rows = h_scaled / 2;
    
    if (tx >= render_cols || ty >= render_rows) {
      sx = sy = -1;
      return;
    }
    
    sx = (tx * capture.width) / w_scaled;
    sy = (ty * 2 * capture.height) / h_scaled;
    
  } else {
    // ASCII
    if (auto_fit) {
      double aspect = (double)capture.width / capture.height;
      double term_aspect = (double)tw / th;
      if (aspect > term_aspect) {
        w_scaled = tw;
        h_scaled = (int)(w_scaled / aspect);
      } else {
        h_scaled = th;
        w_scaled = (int)(h_scaled * aspect);
      }
    } else {
      w_scaled = (int)(capture.width * scale_factor);
      h_scaled = (int)(capture.height * scale_factor);
    }
    w_scaled = std::min(w_scaled, tw);
    h_scaled = std::min(h_scaled, th);
    
    if (tx >= w_scaled || ty >= h_scaled) {
      sx = sy = -1;
      return;
    }
    
    sx = (tx * capture.width) / w_scaled;
    sy = (ty * capture.height) / h_scaled;
  }
  
  // Clamp to screen bounds
  sx = std::max(0, std::min(sx, capture.width - 1));
  sy = std::max(0, std::min(sy, capture.height - 1));
}

// --- Helper to get RGB from pixel data ---
static inline void get_rgb(const uint8_t * p, uint8_t & r, uint8_t & g, uint8_t & b) {
  b = p[0];
  g = p[1];
  r = p[2];
}

static void handle_character(uint8_t c) {
  if (!virtual_keyboard) return;
  std::cerr << "handle_character received: '" << (char)c << "' (0x" << std::hex << (int)c << std::dec << ")" << std::endl;
  
  uint32_t keycode = 0;
  bool shift = false;
  bool ctrl = false;
  
  // Special control characters first (but skip printable ASCII)
  if (c < 0x20 && c != '\t' && c != '\n' && c != '\r') {
    // Ctrl+letter combinations (0x01-0x1A = Ctrl+A through Ctrl+Z)
    if (c >= 0x01 && c <= 0x1A) {
      // Map to appropriate KEY_* constant
      switch(c) {
        case 0x01: keycode = KEY_A; break;
        case 0x02: keycode = KEY_B; break;
        case 0x03: keycode = KEY_C; break;
        case 0x04: keycode = KEY_D; break;
        case 0x05: keycode = KEY_E; break;
        case 0x06: keycode = KEY_F; break;
        case 0x07: keycode = KEY_G; break;
        case 0x08: keycode = KEY_H; break;
        case 0x09: keycode = KEY_I; break;
        case 0x0A: keycode = KEY_J; break;
        case 0x0B: keycode = KEY_K; break;
        case 0x0C: keycode = KEY_L; break;
        case 0x0D: keycode = KEY_M; break;
        case 0x0E: keycode = KEY_N; break;
        case 0x0F: keycode = KEY_O; break;
        case 0x10: keycode = KEY_P; break;
        case 0x11: keycode = KEY_Q; break;
        case 0x12: keycode = KEY_R; break;
        case 0x13: keycode = KEY_S; break;
        case 0x14: keycode = KEY_T; break;
        case 0x15: keycode = KEY_U; break;
        case 0x16: keycode = KEY_V; break;
        case 0x17: keycode = KEY_W; break;
        case 0x18: keycode = KEY_X; break;
        case 0x19: keycode = KEY_Y; break;
        case 0x1A: keycode = KEY_Z; break;
      }
      ctrl = true;
    }
    // Other control chars
    else {
      switch (c) {
        case 0x00: keycode = KEY_SPACE; ctrl = true; break; // Ctrl+Space
        case 0x1C: keycode = KEY_BACKSLASH; ctrl = true; break; // Ctrl+backslash
        case 0x1D: keycode = KEY_RIGHTBRACE; ctrl = true; break; // Ctrl+]
        case 0x1E: keycode = KEY_6; ctrl = true; shift = true; break; // Ctrl+^
        case 0x1F: keycode = KEY_MINUS; ctrl = true; shift = true; break; // Ctrl+_
        default: return; // Unknown control char
      }
    }
  }
  // Regular printable ASCII
  else if (c >= 0x20 && c <= 0x7E) {
    // Letters
    if (c >= 'a' && c <= 'z') {
      switch(c) {
        case 'a': keycode = KEY_A; break;
        case 'b': keycode = KEY_B; break;
        case 'c': keycode = KEY_C; break;
        case 'd': keycode = KEY_D; break;
        case 'e': keycode = KEY_E; break;
        case 'f': keycode = KEY_F; break;
        case 'g': keycode = KEY_G; break;
        case 'h': keycode = KEY_H; break;
        case 'i': keycode = KEY_I; break;
        case 'j': keycode = KEY_J; break;
        case 'k': keycode = KEY_K; break;
        case 'l': keycode = KEY_L; break;
        case 'm': keycode = KEY_M; break;
        case 'n': keycode = KEY_N; break;
        case 'o': keycode = KEY_O; break;
        case 'p': keycode = KEY_P; break;
        case 'q': keycode = KEY_Q; break;
        case 'r': keycode = KEY_R; break;
        case 's': keycode = KEY_S; break;
        case 't': keycode = KEY_T; break;
        case 'u': keycode = KEY_U; break;
        case 'v': keycode = KEY_V; break;
        case 'w': keycode = KEY_W; break;
        case 'x': keycode = KEY_X; break;
        case 'y': keycode = KEY_Y; break;
        case 'z': keycode = KEY_Z; break;
      }
    } else if (c >= 'A' && c <= 'Z') {
      switch(c) {
        case 'A': keycode = KEY_A; break;
        case 'B': keycode = KEY_B; break;
        case 'C': keycode = KEY_C; break;
        case 'D': keycode = KEY_D; break;
        case 'E': keycode = KEY_E; break;
        case 'F': keycode = KEY_F; break;
        case 'G': keycode = KEY_G; break;
        case 'H': keycode = KEY_H; break;
        case 'I': keycode = KEY_I; break;
        case 'J': keycode = KEY_J; break;
        case 'K': keycode = KEY_K; break;
        case 'L': keycode = KEY_L; break;
        case 'M': keycode = KEY_M; break;
        case 'N': keycode = KEY_N; break;
        case 'O': keycode = KEY_O; break;
        case 'P': keycode = KEY_P; break;
        case 'Q': keycode = KEY_Q; break;
        case 'R': keycode = KEY_R; break;
        case 'S': keycode = KEY_S; break;
        case 'T': keycode = KEY_T; break;
        case 'U': keycode = KEY_U; break;
        case 'V': keycode = KEY_V; break;
        case 'W': keycode = KEY_W; break;
        case 'X': keycode = KEY_X; break;
        case 'Y': keycode = KEY_Y; break;
        case 'Z': keycode = KEY_Z; break;
      }
      shift = true;
    }
    // Numbers
    else if (c >= '0' && c <= '9') {
      switch(c) {
        case '0': keycode = KEY_0; break;
        case '1': keycode = KEY_1; break;
        case '2': keycode = KEY_2; break;
        case '3': keycode = KEY_3; break;
        case '4': keycode = KEY_4; break;
        case '5': keycode = KEY_5; break;
        case '6': keycode = KEY_6; break;
        case '7': keycode = KEY_7; break;
        case '8': keycode = KEY_8; break;
        case '9': keycode = KEY_9; break;
      }
    }
    // Punctuation and symbols
    else {
      switch (c) {
        case ' ': keycode = KEY_SPACE; break;
        
        // Shifted number row
        case '!': keycode = KEY_1; shift = true; break;
        case '@': keycode = KEY_2; shift = true; break;
        case '#': keycode = KEY_3; shift = true; break;
        case '$': keycode = KEY_4; shift = true; break;
        case '%': keycode = KEY_5; shift = true; break;
        case '^': keycode = KEY_6; shift = true; break;
        case '&': keycode = KEY_7; shift = true; break;
        case '*': keycode = KEY_8; shift = true; break;
        case '(': keycode = KEY_9; shift = true; break;
        case ')': keycode = KEY_0; shift = true; break;
        
        // Punctuation
        case '-': keycode = KEY_MINUS; break;
        case '_': keycode = KEY_MINUS; shift = true; break;
        case '=': keycode = KEY_EQUAL; break;
        case '+': keycode = KEY_EQUAL; shift = true; break;
        case '[': keycode = KEY_LEFTBRACE; break;
        case '{': keycode = KEY_LEFTBRACE; shift = true; break;
        case ']': keycode = KEY_RIGHTBRACE; break;
        case '}': keycode = KEY_RIGHTBRACE; shift = true; break;
        case '\\': keycode = KEY_BACKSLASH; break;
        case '|': keycode = KEY_BACKSLASH; shift = true; break;
        case ';': keycode = KEY_SEMICOLON; break;
        case ':': keycode = KEY_SEMICOLON; shift = true; break;
        case '\'': keycode = KEY_APOSTROPHE; break;
        case '"': keycode = KEY_APOSTROPHE; shift = true; break;
        case ',': keycode = KEY_COMMA; break;
        case '<': keycode = KEY_COMMA; shift = true; break;
        case '.': keycode = KEY_DOT; break;
        case '>': keycode = KEY_DOT; shift = true; break;
        case '/': keycode = KEY_SLASH; break;
        case '?': keycode = KEY_SLASH; shift = true; break;
        case '`': keycode = KEY_GRAVE; break;
        case '~': keycode = KEY_GRAVE; shift = true; break;
        
        default: return; // Unsupported character
      }
    }
  }
  // Special keys that have their own codes
  else {
    switch (c) {
      case '\n': case '\r': keycode = KEY_ENTER; break;
      case '\t': keycode = KEY_TAB; break;
      case '\b': case 127: keycode = KEY_BACKSPACE; break;
      case 27: keycode = KEY_ESC; break;
      default: return; // Unsupported
    }
  }
  
  if (keycode) {
    send_key(keycode, shift, ctrl);
  }
}

// Also update handle_escape_sequence to support Ctrl+Arrow keys and other combinations
static bool handle_escape_sequence(const uint8_t* buf, int len, int& consumed) {
  if (len < 2 || buf[0] != 0x1b) return false;
  
  // Check for double escape (ESC key)
  if (buf[1] == 0x1b) {
    send_key(KEY_ESC);
    consumed = 2;
    return true;
  }
  
  // Arrow keys and special keys
  if (len >= 3 && buf[1] == '[') {
    uint32_t keycode = 0;
    bool ctrl = false;
    bool shift = false;
    consumed = 3;
    
    // Check for modifiers (ESC[1;xA format where x is modifier)
    if (len >= 6 && buf[2] == '1' && buf[3] == ';') {
      int mod = buf[4] - '0';
      // Modifier encoding: 2=Shift, 3=Alt, 4=Shift+Alt, 5=Ctrl, 6=Shift+Ctrl, 7=Alt+Ctrl, 8=Shift+Alt+Ctrl
      shift = (mod == 2 || mod == 4 || mod == 6 || mod == 8);
      ctrl = (mod == 5 || mod == 6 || mod == 7 || mod == 8);
      
      switch (buf[5]) {
        case 'A': keycode = KEY_UP; break;
        case 'B': keycode = KEY_DOWN; break;
        case 'C': keycode = KEY_RIGHT; break;
        case 'D': keycode = KEY_LEFT; break;
        case 'H': keycode = KEY_HOME; break;
        case 'F': keycode = KEY_END; break;
      }
      consumed = 6;
    }
    // Standard arrow keys without modifiers
    else {
      switch (buf[2]) {
        case 'A': keycode = KEY_UP; break;
        case 'B': keycode = KEY_DOWN; break;
        case 'C': keycode = KEY_RIGHT; break;
        case 'D': keycode = KEY_LEFT; break;
        case 'H': keycode = KEY_HOME; break;
        case 'F': keycode = KEY_END; break;
        
        // Handle extended sequences (e.g., Page Up/Down, Delete)
        case '1': case '2': case '3': case '4': case '5': case '6':
          if (len >= 4 && buf[3] == '~') {
            switch (buf[2]) {
              case '1': keycode = KEY_HOME; break;
              case '2': keycode = KEY_INSERT; break;
              case '3': keycode = KEY_DELETE; break;
              case '4': keycode = KEY_END; break;
              case '5': keycode = KEY_PAGEUP; break;
              case '6': keycode = KEY_PAGEDOWN; break;
            }
            consumed = 4;
          }
          break;
        
        default:
          return false; // Unknown sequence
      }
    }
    
    if (keycode) {
      send_key(keycode, shift, ctrl);
      return true;
    }
  }
  
  return false;
}

// Improved mojjje input handling with proper protocol usage
static void handle_mouse_input(int button, int x, int y, bool pressed) {
  if (!virtual_pointer) return;
  
  mouse_x = x - 1;
  mouse_y = y - 1;
  
  int screen_x, screen_y;
  terminal_to_screen_coords(mouse_x, mouse_y, screen_x, screen_y);
  
  if (screen_x == -1 || screen_y == -1) {
    return;
  }

  // Use wl_fixed_t for proper coordinate precision
  wl_fixed_t fixed_x = wl_fixed_from_int(screen_x);
  wl_fixed_t fixed_y = wl_fixed_from_int(screen_y);

  // Move pointer to absolute position with proper API
  zwlr_virtual_pointer_v1_motion_absolute(virtual_pointer, 
    0, // time - 0 for immediate
    fixed_x, 
    fixed_y,
    capture.width, 
    capture.height);
  
  // Handle button with proper state enum
  if (button >= 1 && button <= 3) {
    uint32_t linux_button = BTN_LEFT;
    if (button == 2) linux_button = BTN_MIDDLE;
    else if (button == 3) linux_button = BTN_RIGHT;
    
    zwlr_virtual_pointer_v1_button(virtual_pointer, 
      0, // time
      linux_button,
      pressed ? WL_POINTER_BUTTON_STATE_PRESSED : WL_POINTER_BUTTON_STATE_RELEASED);
  }
  
  // Frame must be called to commit the events
  zwlr_virtual_pointer_v1_frame(virtual_pointer);
  wl_display_flush(display);
}

// Add scroll wheel support
static void handle_scroll_input(int direction) {
  if (!virtual_pointer) return;
  
  // direction: 1 = up, -1 = down
  wl_fixed_t value = wl_fixed_from_int(direction * 15); // 15 pixels per scroll
  
  zwlr_virtual_pointer_v1_axis(virtual_pointer,
    0, // time
    WL_POINTER_AXIS_VERTICAL_SCROLL,
    value);
  
  zwlr_virtual_pointer_v1_frame(virtual_pointer);
  wl_display_flush(display);
}

// Update parse_mouse_sgr to handle scroll
static bool parse_mouse_sgr(const char* buf, int len, int& button, int& x, int& y, bool& pressed, bool& is_scroll) {
  if (len < 6 || buf[0] != '\033' || buf[1] != '[' || buf[2] != '<') return false;
  
  int b = 0, xpos = 0, ypos = 0;
  int i = 3;
  
  while (i < len && buf[i] >= '0' && buf[i] <= '9') {
    b = b * 10 + (buf[i] - '0');
    i++;
  }
  if (i >= len || buf[i] != ';') return false;
  i++;
  
  while (i < len && buf[i] >= '0' && buf[i] <= '9') {
    xpos = xpos * 10 + (buf[i] - '0');
    i++;
  }
  if (i >= len || buf[i] != ';') return false;
  i++;
  
  while (i < len && buf[i] >= '0' && buf[i] <= '9') {
    ypos = ypos * 10 + (buf[i] - '0');
    i++;
  }
  if (i >= len) return false;
  
  pressed = (buf[i] == 'M');
  
  // Check for scroll wheel (button codes 64-65)
  is_scroll = (b >= 64 && b <= 65);
  if (is_scroll) {
    button = (b == 64) ? 1 : -1; // 1 for up, -1 for down
  } else {
    button = (b & 0x03) + 1;
  }
  
  x = xpos;
  y = ypos;
  
  return true;
}

// Update process_input to ignore input while sending
static void process_input() {
  if (!input_enabled || sending_input) return;  // Skip if we're sending
  
  uint8_t buf[256];
  int n = read(STDIN_FILENO, buf, sizeof(buf));
  if (n <= 0) return;
  
  // Flush stdin to prevent buildup
  tcflush(STDIN_FILENO, TCIFLUSH);
  
  // Debug: log received bytes
  std::cerr << "Received " << n << " bytes: ";
  for (int i = 0; i < n; i++) {
    std::cerr << std::hex << (int)buf[i] << " ";
  }
  std::cerr << std::dec << std::endl;
  
  // Check for quit sequence: ESC ESC Ctrl-X
  for (int i = 0; i < n - 2; i++) {
    if (buf[i] == 0x1b && buf[i+1] == 0x1b && buf[i+2] == 0x18) {
      done = true;
      return;
    }
  }
  
  // Check for mouse input (SGR format: ESC[<b;x;yM or m)
  if (n >= 6 && buf[0] == '\033' && buf[1] == '[' && buf[2] == '<') {
    int button, x, y;
    bool pressed, is_scroll = false;
    if (parse_mouse_sgr((const char*)buf, n, button, x, y, pressed, is_scroll)) {
      if (is_scroll) {
        handle_scroll_input(button);
      } else {
        handle_mouse_input(button, x, y, pressed);
      }
      return;
    }
  }
  
  // Process keyboard input
  int i = 0;
  while (i < n) {
    int consumed = 0;
    
    // Try escape sequences first
    if (buf[i] == 0x1b && i + 1 < n && handle_escape_sequence(buf + i, n - i, consumed)) {
      i += consumed;
      continue;
    }
    
    // Handle single character
    handle_character(buf[i]);
    i++;
  }
}

// Better keymap setup with validation
static void setup_virtual_keyboard_keymap(const std::string& layout) {
  if (!virtual_keyboard) return;
  
  // Build proper XKB keymap string
  std::string keymap_str = 
    "xkb_keymap {\n"
    "  xkb_keycodes { include \"evdev+aliases(qwerty)\" };\n"
    "  xkb_types { include \"complete\" };\n"
    "  xkb_compat { include \"complete\" };\n"
    "  xkb_symbols { include \"pc+"+layout+"+inet(evdev)\" };\n"
    "  xkb_geometry { include \"pc(pc105)\" };\n"
    "};\n";
  
  size_t keymap_size = keymap_str.size();
  int fd = create_shm_file(keymap_size);
  if (fd < 0) {
    std::cerr << "Failed to create keymap file\n";
    return;
  }
  
  void* data = mmap(nullptr, keymap_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (data == MAP_FAILED) {
    close(fd);
    std::cerr << "Failed to mmap keymap\n";
    return;
  }
  
  memcpy(data, keymap_str.c_str(), keymap_size);
  munmap(data, keymap_size);
  
  // Send keymap to compositor
  zwp_virtual_keyboard_v1_keymap(virtual_keyboard, 
    WL_KEYBOARD_KEYMAP_FORMAT_XKB_V1, 
    fd, 
    keymap_size);
  
  close(fd);
  wl_display_flush(display);
  
  std::cerr << "Virtual keyboard initialized with layout: " << layout << "\n";
}

static void render_blocks_scaled() {
  const uint8_t * pix = static_cast<const uint8_t *>(capture.data);
  int tw, th;
  get_terminal_size(tw, th);
  
  int w_scaled, h_scaled;
  if (auto_fit) {
    // Fit to terminal while preserving aspect ratio
    double aspect = (double)capture.width / capture.height;
    double term_aspect = (double)tw / (th * 2); // blocks are 1x2
    if (aspect > term_aspect) {
      w_scaled = tw;
      h_scaled = (int)(w_scaled / aspect);
    } else {
      h_scaled = th * 2;
      w_scaled = (int)(h_scaled * aspect);
    }
  } else {
    w_scaled = (int)(capture.width * scale_factor);
    h_scaled = (int)(capture.height * scale_factor);
  }
  
  // Clamp to terminal size
  w_scaled = std::min(w_scaled, tw);
  h_scaled = std::min(h_scaled, th * 2);
  
  for (int y = 0; y < h_scaled / 2; y++) {
    for (int x = 0; x < w_scaled; x++) {
      int src_x = x * capture.width / w_scaled;
      int src_y_top = (y * 2) * capture.height / h_scaled;
      int src_y_bot = (y * 2 + 1) * capture.height / h_scaled;
      
      src_x = std::min(src_x, capture.width - 1);
      src_y_top = std::min(src_y_top, capture.height - 1);
      src_y_bot = std::min(src_y_bot, capture.height - 1);
      
      const uint8_t * top = pix + src_y_top * capture.stride + src_x * 4;
      const uint8_t * bot = pix + src_y_bot * capture.stride + src_x * 4;
      
      uint8_t r_top, g_top, b_top, r_bot, g_bot, b_bot;
      get_rgb(top, r_top, g_top, b_top);
      get_rgb(bot, r_bot, g_bot, b_bot);
      
      std::string fg = rgb_to_ansi(r_top, g_top, b_top, mode), bg;
      switch (mode) {
      case ColorMode::ANSI_16:
        bg = "\033[48;5;" + std::to_string(rgb_to_ansi_16(r_bot, g_bot, b_bot)) + "m";
        break;
      case ColorMode::ANSI_256:
        bg = "\033[48;5;" + std::to_string(rgb_to_ansi_256(r_bot, g_bot, b_bot)) + "m";
        break;
      case ColorMode::ANSI_TRUECOLOR:
        bg = "\033[48;2;" + std::to_string(r_bot) + ";" + std::to_string(g_bot) + ";" + std::to_string(b_bot) + "m";
        break;
      }
      std::cout << bg << fg << "▀";
    }
    std::cout << "\033[0m";

    if (y < h_scaled / 2 - 1) {
      std::cout << '\n';
    }
  }
}

static void render_ascii_scaled() {
  const uint8_t * pix = static_cast <
    const uint8_t * > (capture.data);
  const char * chars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
  int tw, th;
  get_terminal_size(tw, th);
  
  int w_scaled, h_scaled;
  if (auto_fit) {
    // Fit to terminal while preserving aspect ratio
    double aspect = (double)capture.width / capture.height;
    double term_aspect = (double)tw / th;
    if (aspect > term_aspect) {
      w_scaled = tw;
      h_scaled = (int)(w_scaled / aspect);
    } else {
      h_scaled = th;
      w_scaled = (int)(h_scaled * aspect);
    }
  } else {
    w_scaled = (int)(capture.width * scale_factor);
    h_scaled = (int)(capture.height * scale_factor);
  }
  
  // Clamp to terminal size
  w_scaled = std::min(w_scaled, tw);
  h_scaled = std::min(h_scaled, th);
  
  for (int y = 0; y < h_scaled; y++) {
    for (int x = 0; x < w_scaled; x++) {
      int src_x = x * capture.width / w_scaled;
      int src_y = y * capture.height / h_scaled;
      src_x = std::min(src_x, capture.width - 1);
      src_y = std::min(src_y, capture.height - 1);
      
      const uint8_t * p = pix + src_y * capture.stride + src_x * 4;
      uint8_t r, g, b;
      get_rgb(p, r, g, b);
      double luma = 0.299 * r + 0.587 * g + 0.114 * b;
      int idx = std::min((int)((luma / 255.0) * (strlen(chars) - 1)), (int) strlen(chars) - 1);
      std::cout << rgb_to_ansi(r, g, b, mode) << chars[idx];
    }
    std::cout << "\033[0m";

    if (y < h_scaled - 1) {
      std::cout << '\n';
    }
  }
}

static double calculate_local_variance(const uint8_t* pix, int x, int y, int w, int h, int stride) {
  double sum = 0, sum_sq = 0;
  int count = 0;
  
  // Sample 2x4 area (braille size)
  for (int dy = 0; dy < 4 && y + dy < h; dy++) {
    for (int dx = 0; dx < 2 && x + dx < w; dx++) {
      int px = std::min(x + dx, w - 1);
      int py = std::min(y + dy, h - 1);
      const uint8_t* p = pix + py * stride + px * 4;
      uint8_t r, g, b;
      get_rgb(p, r, g, b);
      double luma = 0.299 * r + 0.587 * g + 0.114 * b;
      sum += luma;
      sum_sq += luma * luma;
      count++;
    }
  }
  
  if (count == 0) return 0;
  double mean = sum / count;
  double variance = (sum_sq / count) - (mean * mean);
  return variance;
}

static void render_braille_scaled() {
  const uint8_t * pix = static_cast<const uint8_t*>(capture.data);
  get_terminal_size(term_width, term_height);
  
  int w_scaled, h_scaled;
  if (auto_fit) {
    double aspect = (double)capture.width / capture.height;
    double term_aspect = (double)(term_width * 2) / (term_height * 4);
    if (aspect > term_aspect) {
      w_scaled = term_width * 2;
      h_scaled = (int)(w_scaled / aspect);
    } else {
      h_scaled = term_height * 4;
      w_scaled = (int)(h_scaled * aspect);
    }
  } else {
    w_scaled = (int)(capture.width * scale_factor);
    h_scaled = (int)(capture.height * scale_factor);
  }
  
  w_scaled = std::min(w_scaled, term_width * 2);
  h_scaled = std::min(h_scaled, term_height * 4);
  
  for (int y = 0; y < h_scaled / 4; y++) {
    for (int x = 0; x < w_scaled / 2; x++) {
      uint8_t pattern = 0;
      uint32_t r_sum = 0, g_sum = 0, b_sum = 0;
      uint32_t r_fg = 0, g_fg = 0, b_fg = 0;
      uint32_t r_bg = 0, g_bg = 0, b_bg = 0;
      int count_fg = 0, count_bg = 0;
      double lumas[8];
      int luma_idx = 0;
      
      // First pass: collect all colors and lumas
      for (int dy = 0; dy < 4; dy++) {
        for (int dx = 0; dx < 2; dx++) {
          int px = (x * 2 + dx) * capture.width / w_scaled;
          int py = (y * 4 + dy) * capture.height / h_scaled;
          px = std::min(px, capture.width - 1);
          py = std::min(py, capture.height - 1);
          
          const uint8_t * p = pix + py * capture.stride + px * 4;
          uint8_t r, g, b;
          get_rgb(p, r, g, b);
          
          r_sum += r; g_sum += g; b_sum += b;
          lumas[luma_idx++] = 0.299 * r + 0.587 * g + 0.114 * b;
        }
      }
      
      // Calculate adaptive threshold based on local contrast
      double luma_avg = 0;
      for (int i = 0; i < 8; i++) luma_avg += lumas[i];
      luma_avg /= 8.0;
      
      double luma_variance = 0;
      for (int i = 0; i < 8; i++) {
        double diff = lumas[i] - luma_avg;
        luma_variance += diff * diff;
      }
      luma_variance /= 8.0;
      
      // Use adaptive threshold: higher variance = use average as threshold
      // Lower variance = stick closer to a fixed threshold
      double threshold = (luma_variance > 1000) ? luma_avg : 
                        (luma_avg * 0.3 + 96 * 0.7); // Blend average with fixed threshold
      
      // Second pass: classify pixels and accumulate colors
      luma_idx = 0;
      for (int dy = 0; dy < 4; dy++) {
        for (int dx = 0; dx < 2; dx++) {
          int px = (x * 2 + dx) * capture.width / w_scaled;
          int py = (y * 4 + dy) * capture.height / h_scaled;
          px = std::min(px, capture.width - 1);
          py = std::min(py, capture.height - 1);
          
          const uint8_t * p = pix + py * capture.stride + px * 4;
          uint8_t r, g, b;
          get_rgb(p, r, g, b);
          
          if (lumas[luma_idx++] > threshold) {
            r_fg += r; g_fg += g; b_fg += b;
            count_fg++;
            
            int idx = dy * 2 + dx;
            static const uint8_t map[8] = {0x01, 0x08, 0x02, 0x10, 0x04, 0x20, 0x40, 0x80};
            pattern |= map[idx];
          } else {
            r_bg += r; g_bg += g; b_bg += b;
            count_bg++;
          }
        }
      }

      // Set foreground color with boost for better visibility
      const double boost_factor = 1.5;
      if (count_fg > 0) {
        r_fg = std::min((uint32_t)(r_fg * boost_factor), 255u * count_fg);
        g_fg = std::min((uint32_t)(g_fg * boost_factor), 255u * count_fg);
        b_fg = std::min((uint32_t)(b_fg * boost_factor), 255u * count_fg);
        std::cout << rgb_to_ansi(r_fg / count_fg, g_fg / count_fg, b_fg / count_fg, mode);
      } else {
        // All pixels are background - use average color
        std::cout << rgb_to_ansi(r_sum / 8, g_sum / 8, b_sum / 8, mode);
      }
      
      // Set background color - always use meaningful color
      std::string bg;
      uint8_t bg_r, bg_g, bg_b;
      if (count_bg > 0) {
        // Use average of darker pixels
        bg_r = r_bg / count_bg;
        bg_g = g_bg / count_bg;
        bg_b = b_bg / count_bg;
      } else {
        // All pixels are bright - use average of all pixels as background
        bg_r = r_sum / 8;
        bg_g = g_sum / 8;
        bg_b = b_sum / 8;
      }
      
      switch (mode) {
      case ColorMode::ANSI_16:
        bg = "\033[48;5;" + std::to_string(rgb_to_ansi_16(bg_r, bg_g, bg_b)) + "m";
        break;
      case ColorMode::ANSI_256:
        bg = "\033[48;5;" + std::to_string(rgb_to_ansi_256(bg_r, bg_g, bg_b)) + "m";
        break;
      case ColorMode::ANSI_TRUECOLOR:
        bg = "\033[48;2;" + std::to_string(bg_r) + ";" + std::to_string(bg_g) + ";" + std::to_string(bg_b) + "m";
        break;
      }
      std::cout << bg;
      
      int codepoint = 0x2800 + pattern;
      std::cout << (char)(0xE0 | (codepoint >> 12))
                << (char)(0x80 | ((codepoint >> 6) & 0x3F))
                << (char)(0x80 | (codepoint & 0x3F));
    }
    std::cout << "\033[0m";

    if (y < h_scaled / 4 - 1) {
      std::cout << '\n';
    }
  }
}

static void render_hybrid_scaled() {
  const uint8_t* pix = static_cast<const uint8_t*>(capture.data);
  int tw, th;
  get_terminal_size(tw, th);
  
  int w_scaled, h_scaled;
  if (auto_fit) {
    // Fit to terminal while preserving aspect ratio
    double aspect = (double)capture.width / capture.height;
    double term_aspect = (double)(tw * 2) / (th * 4); // braille is 2x4
    if (aspect > term_aspect) {
      w_scaled = tw * 2;
      h_scaled = (int)(w_scaled / aspect);
    } else {
      h_scaled = th * 4;
      w_scaled = (int)(h_scaled * aspect);
    }
  } else {
    w_scaled = (int)(capture.width * scale_factor);
    h_scaled = (int)(capture.height * scale_factor);
  }
  
  // Clamp to terminal size
  w_scaled = std::min(w_scaled, tw * 2);
  h_scaled = std::min(h_scaled, th * 4);
  
  // Variance threshold for switching between braille and blocks
  const double VARIANCE_THRESHOLD = 100.0; // Tune it ig
  
  for (int y = 0; y < h_scaled / 4; y++) {
    for (int x = 0; x < w_scaled / 2; x++) {
      // Map to source coordinates
      int src_x = (x * 2) * capture.width / w_scaled;
      int src_y = (y * 4) * capture.height / h_scaled;
      src_x = std::min(src_x, capture.width - 1);
      src_y = std::min(src_y, capture.height - 1);
      
      // Calculate local detail
      double variance = calculate_local_variance(pix, src_x, src_y, 
                                                  capture.width, capture.height, 
                                                  capture.stride);
      
      if (variance > VARIANCE_THRESHOLD) {
        // High detail area - use braille with improved color handling
        uint8_t pattern = 0;
        uint32_t r_sum = 0, g_sum = 0, b_sum = 0;
        uint32_t r_fg = 0, g_fg = 0, b_fg = 0;
        uint32_t r_bg = 0, g_bg = 0, b_bg = 0;
        int count_fg = 0, count_bg = 0;
        double lumas[8];
        int luma_idx = 0;
        
        // First pass: collect all colors and lumas
        for (int dy = 0; dy < 4; dy++) {
          for (int dx = 0; dx < 2; dx++) {
            int px = (x * 2 + dx) * capture.width / w_scaled;
            int py = (y * 4 + dy) * capture.height / h_scaled;
            px = std::min(px, capture.width - 1);
            py = std::min(py, capture.height - 1);
            
            const uint8_t* p = pix + py * capture.stride + px * 4;
            uint8_t r, g, b;
            get_rgb(p, r, g, b);
            
            r_sum += r; g_sum += g; b_sum += b;
            lumas[luma_idx++] = 0.299 * r + 0.587 * g + 0.114 * b;
          }
        }
        
        // Calculate adaptive threshold
        double luma_avg = 0;
        for (int i = 0; i < 8; i++) luma_avg += lumas[i];
        luma_avg /= 8.0;
        
        double luma_variance = 0;
        for (int i = 0; i < 8; i++) {
          double diff = lumas[i] - luma_avg;
          luma_variance += diff * diff;
        }
        luma_variance /= 8.0;
        
        double threshold = (luma_variance > 1000) ? luma_avg : 
                          (luma_avg * 0.3 + 96 * 0.7);
        
        // Second pass: classify pixels and accumulate colors
        luma_idx = 0;
        for (int dy = 0; dy < 4; dy++) {
          for (int dx = 0; dx < 2; dx++) {
            int px = (x * 2 + dx) * capture.width / w_scaled;
            int py = (y * 4 + dy) * capture.height / h_scaled;
            px = std::min(px, capture.width - 1);
            py = std::min(py, capture.height - 1);
            
            const uint8_t* p = pix + py * capture.stride + px * 4;
            uint8_t r, g, b;
            get_rgb(p, r, g, b);
            
            if (lumas[luma_idx++] > threshold) {
              r_fg += r; g_fg += g; b_fg += b;
              count_fg++;
              
              int idx = dy * 2 + dx;
              static const uint8_t map[8] = {0x01, 0x08, 0x02, 0x10, 0x04, 0x20, 0x40, 0x80};
              pattern |= map[idx];
            } else {
              r_bg += r; g_bg += g; b_bg += b;
              count_bg++;
            }
          }
        }
        
        // Set foreground color with boost
        const double boost_factor = 1.5;
        if (count_fg > 0) {
          r_fg = std::min((uint32_t)(r_fg * boost_factor), 255u * count_fg);
          g_fg = std::min((uint32_t)(g_fg * boost_factor), 255u * count_fg);
          b_fg = std::min((uint32_t)(b_fg * boost_factor), 255u * count_fg);
          std::cout << rgb_to_ansi(r_fg / count_fg, g_fg / count_fg, b_fg / count_fg, mode);
        } else {
          // All pixels are background - use average color
          std::cout << rgb_to_ansi(r_sum / 8, g_sum / 8, b_sum / 8, mode);
        }
        
        // Set background color - always use meaningful color
        std::string bg;
        uint8_t bg_r, bg_g, bg_b;
        if (count_bg > 0) {
          bg_r = r_bg / count_bg;
          bg_g = g_bg / count_bg;
          bg_b = b_bg / count_bg;
        } else {
          // All pixels are bright - use average of all pixels
          bg_r = r_sum / 8;
          bg_g = g_sum / 8;
          bg_b = b_sum / 8;
        }
        
        switch (mode) {
        case ColorMode::ANSI_16:
          bg = "\033[48;5;" + std::to_string(rgb_to_ansi_16(bg_r, bg_g, bg_b)) + "m";
          break;
        case ColorMode::ANSI_256:
          bg = "\033[48;5;" + std::to_string(rgb_to_ansi_256(bg_r, bg_g, bg_b)) + "m";
          break;
        case ColorMode::ANSI_TRUECOLOR:
          bg = "\033[48;2;" + std::to_string(bg_r) + ";" + std::to_string(bg_g) + ";" + std::to_string(bg_b) + "m";
          break;
        }
        std::cout << bg;
        
        // Output UTF-8 braille character
        int codepoint = 0x2800 + pattern;
        std::cout << (char)(0xE0 | (codepoint >> 12))
                  << (char)(0x80 | ((codepoint >> 6) & 0x3F))
                  << (char)(0x80 | (codepoint & 0x3F));
      } else {
        // Low detail area - use blocks (unchanged, blocks already handle colors well)
        int src_y_top = (y * 4) * capture.height / h_scaled;
        int src_y_mid = (y * 4 + 2) * capture.height / h_scaled;
        
        src_y_top = std::min(src_y_top, capture.height - 1);
        src_y_mid = std::min(src_y_mid, capture.height - 1);
        
        const uint8_t* top = pix + src_y_top * capture.stride + src_x * 4;
        const uint8_t* bot = pix + src_y_mid * capture.stride + src_x * 4;
        
        uint8_t r_top, g_top, b_top, r_bot, g_bot, b_bot;
        get_rgb(top, r_top, g_top, b_top);
        get_rgb(bot, r_bot, g_bot, b_bot);
        
        std::string fg = rgb_to_ansi(r_top, g_top, b_top, mode), bg;
        switch (mode) {
        case ColorMode::ANSI_16:
          bg = "\033[48;5;" + std::to_string(rgb_to_ansi_16(r_bot, g_bot, b_bot)) + "m";
          break;
        case ColorMode::ANSI_256:
          bg = "\033[48;5;" + std::to_string(rgb_to_ansi_256(r_bot, g_bot, b_bot)) + "m";
          break;
        case ColorMode::ANSI_TRUECOLOR:
          bg = "\033[48;2;" + std::to_string(r_bot) + ";" + std::to_string(g_bot) + ";" + std::to_string(b_bot) + "m";
          break;
        }
        std::cout << bg << fg << "▀";
      }
    }
    std::cout << "\033[0m";

    if (y < h_scaled / 4 - 1) {
      std::cout << '\n';
    }
  }
}

// --- Main ---
int main(int argc, char ** argv) {
  argparse::ArgumentParser program("wayterm_mirror");
  program.add_argument("--mode").default_value("256").help("Color mode: 16, 256, true");
  program.add_argument("--renderer").default_value("braille").help("Renderer: braille");
  program.add_argument("--scale").default_value(1.0).scan<'g', double>().help("Scale factor");
  program.add_argument("--fit").default_value(false).implicit_value(true).help("Auto-fit to terminal");
  program.add_argument("--fps").default_value(10).scan<'i', int>().help("Frames per second");
  program.add_argument("--output").default_value("").help("Output index");
  program.add_argument("--input").default_value(false).implicit_value(true).help("Enable input forwarding");
  program.add_argument("--keymap").default_value("pl").help("Keyboard layout (e.g., pl, us, de)");

  try {
    program.parse_args(argc, argv);
  } catch (...) {
    std::cerr << program;
    return 1;
  }

  std::string m = program.get<std::string>("--mode");
  if (m == "16") mode = ColorMode::ANSI_16;
  else if (m == "256") mode = ColorMode::ANSI_256;
  else mode = ColorMode::ANSI_TRUECOLOR;
  
  renderer = program.get<std::string>("--renderer");
  scale_factor = program.get<double>("--scale");
  auto_fit = program.get<bool>("--fit");
  input_enabled = program.get<bool>("--input");
  
  std::string output_arg = program.get<std::string>("--output");
  if (!output_arg.empty()) {
    selected_output_index = std::stoi(output_arg);
  }

  display = wl_display_connect(nullptr);
  if (!display) {
    std::cerr << "Failed to connect to Wayland\n";
    return 1;
  }
  
  registry = wl_display_get_registry(display);
  wl_registry_add_listener(registry, &registry_listener, nullptr);
  wl_display_roundtrip(display);
  
  if (!shm || outputs.empty() || !manager) {
    std::cerr << "Missing Wayland interfaces\n";
    return 1;
  }
  
  if (selected_output_index < 0 || selected_output_index >= (int)outputs.size()) {
    std::cerr << "Invalid output index\n";
    return 1;
  }
  
  output = outputs[selected_output_index];
  
  // Initialize input devices if enabled
  if (input_enabled) {
    if (!seat) {
      std::cerr << "Warning: No seat found, input forwarding disabled\n";
      input_enabled = false;
    } else {
      if (pointer_manager) {
        virtual_pointer = zwlr_virtual_pointer_manager_v1_create_virtual_pointer(pointer_manager, seat);
      } else {
        std::cerr << "Warning: Virtual pointer not supported\n";
      }
      
      std::string keymap_layout = program.get<std::string>("--keymap");

      // ... later in the input initialization section:

      if (keyboard_manager) {
        virtual_keyboard = zwp_virtual_keyboard_manager_v1_create_virtual_keyboard(keyboard_manager, seat);
        setup_virtual_keyboard_keymap(keymap_layout);  // Pass the layout
      } else {
        std::cerr << "Warning: Virtual keyboard not supported\n";
      }

      enable_raw_mode();
      // DONT FUCKING CHANGE THIS LINE
      std::cerr << "Input forwarding enabled. Press 'ctrl+alt+shift+x' to quit.\n";
    }
  }

  const long double frame_delay = 1000000 / program.get<int>("--fps");

  while (true) {
    auto bg_time = std::chrono::high_resolution_clock::now();
    
    done = false;
    auto * frame = zwlr_screencopy_manager_v1_capture_output(manager, 1, output);
    zwlr_screencopy_frame_v1_add_listener(frame, &frame_listener, nullptr);
    wl_display_flush(display);

    while (!done) {
      if (wl_display_dispatch(display) == -1) {
        std::cerr << "Display dispatch failed\n";
        return 1;
      }
    }
    
    std::cout << "\033[?25l\033[H";
    if (renderer == "braille") {
      render_braille_scaled();
    }
    else if (renderer == "blocks") {
      render_blocks_scaled();
    }
    else if (renderer == "hybrid") {
      render_hybrid_scaled();
    }
    else {
      render_ascii_scaled();
    }
    std::cout << std::flush;
    
    // Process input ONCE per frame, with feedback prevention
    if (input_enabled && !sending_input) {
      process_input();
      // Clear any remaining input to prevent buildup
      tcflush(STDIN_FILENO, TCIFLUSH);
    }
    
    auto nd_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(nd_time - bg_time).count();
    if (elapsed < frame_delay) {
      usleep(frame_delay - elapsed);
    }
  }

  if (input_enabled) {
    disable_raw_mode();
  }
  
  munmap(capture.data, capture.size);
  wl_buffer_destroy(capture.buffer);
  wl_display_disconnect(display);
  return 0;
}
