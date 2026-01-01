#include <wayland-client.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <memory>
#include <queue>
#include <signal.h>
#include <linux/input-event-codes.h>
#include <argparse/argparse.hpp>
#include <random>
#include <sstream>
#include <iomanip>
#include <lz4.h>
#include <lz4hc.h>
#include <unistd.h>
#include "wlr-screencopy-unstable-v1-client-protocol.h"
#include "virtual-keyboard-unstable-v1-client-protocol.h"
#include "wlr-virtual-pointer-unstable-v1-client-protocol.h"
#include "wlr-foreign-toplevel-management-unstable-v1-client-protocol.h"
#include "pipewire_capture.h"
#include "virtual_input.h"
#include <pipewire-0.3/pipewire/pipewire.h>
#include <spa-0.2/spa/param/audio/format-utils.h>
#include <spa-0.2/spa/param/props.h>
#include <cstdlib>
#include <sstream>
#include <cstdint>
#include <fstream>
#include <rapidjson/document.h>

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
    uint8_t detail_level,
    uint8_t threshold_steps,
    uint8_t *patterns,
    uint8_t *fg_colors,
    uint8_t *bg_colors);

extern "C" void render_hybrid_cuda(
    const uint8_t *frame,
    uint32_t fw, uint32_t fh, uint32_t stride,
    int w_scaled, int h_scaled,
    int cells_x, int cells_y,
    double rotation_angle,
    uint8_t detail, uint8_t threshold_steps,
    uint8_t *modes, // 1=braille, 0=blocks
    uint8_t *patterns,
    uint8_t *fg_colors,
    uint8_t *bg_colors);

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
    uint8_t *fg_colors,
    uint8_t *bg_colors);

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
    uint8_t *intensities,
    uint8_t *fg_colors,
    uint8_t *bg_colors);

// ADD global compositor type:
static std::string compositor_type = "generic";

static std::string exec_command(const char *cmd)
{
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe)
    {
        return "";
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
    {
        result += buffer.data();
    }
    return result;
}

static std::string detect_compositor(const std::string &override = "auto")
{
    if (override != "auto")
    {
        std::cerr << "[FOCUS] Using manual compositor setting: " << override << "\n";
        return override;
    }

    const char *compositor = getenv("XDG_CURRENT_DESKTOP");
    if (!compositor)
    {
        compositor = getenv("WAYLAND_DISPLAY");
    }

    if (compositor)
    {
        std::string comp = compositor;
        std::transform(comp.begin(), comp.end(), comp.begin(), ::tolower);

        if (comp.find("hypr") != std::string::npos)
            return "hyprland";
        if (comp.find("sway") != std::string::npos)
            return "sway";
        if (comp.find("river") != std::string::npos)
            return "river";
        if (comp.find("kde") != std::string::npos || comp.find("plasma") != std::string::npos)
            return "kde";
        if (comp.find("gnome") != std::string::npos)
            return "gnome";
    }

    // std::cerr << "[FOCUS] Unknown compositor, using generic (toplevel protocol only)\n";
    return "generic";
}

// --- Protocol definitions ---
struct ClientConfig
{
    uint32_t output_index;
    uint32_t fps;
    uint32_t term_width;
    uint32_t term_height;
    uint8_t color_mode;
    uint8_t renderer;
    uint8_t keep_aspect_ratio;
    uint8_t compress;
    uint8_t compression_level;
    double scale_factor;
    uint8_t follow_focus;
    uint8_t detail_level;
    uint8_t render_device;
    uint8_t quality;
    double rotation_angle;
};

enum CaptureBackend {
    WLR_SCREENCOPY,  // Original wlr-screencopy protocol
    PIPEWIRE,        // PipeWire screencapture
    AUTO_CAPTURE     // Auto-detect
};

static CaptureBackend capture_backend = AUTO_CAPTURE;
static std::vector<std::unique_ptr<PipeWireCapture>> pipewire_captures;

static VirtualInputManager virtual_input_mgr;

static CaptureBackend detect_capture_backend();

struct SendQueue
{
    std::vector<uint8_t> latest_frame;
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> running{true};
    std::atomic<bool> frame_ready{false};
    std::atomic<int> dropped_frames{0};
};

struct CompressedFrameHeader
{
    uint32_t compressed_size;
    uint32_t uncompressed_size;
};

struct ClientFrameCache
{
    std::string last_full_frame;
    int frames_since_full = 0;
    size_t total_bytes_saved = 0;
    size_t total_deltas_sent = 0;
};

static std::atomic<int> total_frames_dropped{0};
static std::atomic<int> total_frames_sent{0};
static std::atomic<size_t> total_bytes_original{0};
static std::atomic<size_t> total_bytes_compressed{0};

static std::vector<int> all_server_sockets;

struct ZoomConfig
{
    uint8_t enabled;
    uint8_t follow_mouse;
    double zoom_level;
    uint32_t view_width;
    uint32_t view_height;
    int32_t center_x;
    int32_t center_y;
    uint8_t smooth_pan;
    uint32_t pan_speed;
};

struct ZoomState
{
    bool enabled = false;
    bool follow_mouse = true;
    double zoom_level = 2.0;
    uint32_t view_width = 800;
    uint32_t view_height = 600;
    int32_t center_x = 960;
    int32_t center_y = 540;
    int32_t target_center_x = 960; // For smooth panning
    int32_t target_center_y = 540;
    bool smooth_pan = true;
    uint32_t pan_speed = 20;
    uint32_t input_width = 0;
    uint32_t input_height = 0;
    std::mutex mutex;
};

struct ClientConnection
{
    int frame_socket = -1;
    int input_socket = -1;
    int audio_socket = -1;
    int config_socket = -1;
    ClientConfig config;
    std::atomic<bool> active{false};
    std::string client_id;
    ZoomState zoom;
};

static std::map<std::string, std::shared_ptr<ClientConnection>> clients;
static std::mutex clients_mutex;
static std::vector<std::condition_variable> output_cvs;
static std::atomic<int> frames_dropped{0};

struct BrailleCell
{
    double lumas[8];
    uint8_t colors[8][3];
    double weights[8];
    bool has_edge;
    double mean_luma;
    double edge_strength;
};

enum class ColorMode
{
    ANSI_16,
    ANSI_256,
    ANSI_TRUECOLOR
};

enum class MessageType : uint8_t
{
    FRAME_DATA = 1,
    KEY_EVENT = 2,
    MOUSE_MOVE = 3,
    MOUSE_BUTTON = 4,
    MOUSE_SCROLL = 5,
    CLIENT_CONFIG = 6,
    RENDERED_FRAME = 7,
    SCREEN_INFO = 8,
    SESSION_ID = 9,
    COMPRESSED_FRAME = 11,
    DELTA_FRAME = 12,
    AUDIO_DATA = 13,
    AUDIO_FORMAT = 14,
    MICROPHONE_DATA = 15,
    MICROPHONE_FORMAT = 16,
    ZOOM_CONFIG = 17,
    ZOOM_TOGGLE = 18
};

struct AudioFormat
{
    uint32_t sample_rate;
    uint32_t channels;
    uint32_t format; // 0 = S16LE, 1 = F32LE
};

struct AudioDataHeader
{
    uint32_t size;
    uint64_t timestamp_us;
};

struct AudioCapture
{
    pw_thread_loop *loop = nullptr;
    pw_stream *stream = nullptr;
    std::mutex mutex;
    std::queue<std::vector<uint8_t>> audio_queue;
    std::atomic<bool> running{true};
    AudioFormat format{48000, 2, 1};
};

static std::map<std::string, ClientFrameCache> client_frame_cache;
static std::mutex frame_cache_mutex;

static AudioCapture audio_capture;
static int audio_server_socket = -1;
static int config_server_socket = -1;

struct MicrophoneVirtualSource
{
    pw_thread_loop *loop = nullptr;
    pw_stream *stream = nullptr;
    std::mutex mutex;
    std::queue<std::vector<uint8_t>> microphone_queue;
    std::atomic<bool> running{true};
    AudioFormat format{48000, 2, 1};
};

static MicrophoneVirtualSource microphone_virtual_source;
static int microphone_server_socket = -1;

struct DeltaFrameHeader
{
    uint32_t num_changes;
    uint32_t base_frame_size;
};

struct FrameChange
{
    uint32_t offset;
    uint16_t length;
    // Followed by: uint8_t data[length]
};

struct SessionID
{
    char uuid[37]; // UUID string (36 chars + null terminator)
};

struct ScreenInfo
{
    uint32_t width;
    uint32_t height;
};

struct RenderedFrameHeader
{
    uint32_t data_size; // Size of ASCII/ANSI string
};

static std::vector<std::unique_ptr<std::condition_variable>> frame_ready_cvs;

struct FrameHeader
{
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t format;
    uint32_t data_size;
};

struct KeyEvent
{
    uint32_t keycode;
    uint8_t pressed;
    uint8_t shift;
    uint8_t ctrl;
    uint8_t alt;
};

struct MouseMove
{
    int32_t x;
    int32_t y;
    uint32_t width;
    uint32_t height;
};

struct MouseButton
{
    uint32_t button;
    uint8_t pressed;
};

struct MouseScroll
{
    int32_t direction;
};

struct SendBuffer
{
    std::vector<uint8_t> data;
    size_t offset = 0;

    void reset()
    {
        data.clear();
        offset = 0;
    }

    void append(const void *ptr, size_t size)
    {
        const uint8_t *bytes = static_cast<const uint8_t *>(ptr);
        data.insert(data.end(), bytes, bytes + size);
    }

    bool has_data() const
    {
        return offset < data.size();
    }

    size_t remaining() const
    {
        return data.size() - offset;
    }

    const uint8_t *current() const
    {
        return data.data() + offset;
    }

    void advance(size_t n)
    {
        offset += n;
        // Compact buffer periodically
        if (offset > 1024 * 1024 && offset == data.size())
        {
            reset();
        }
    }
};

// --- Wayland globals ---
static wl_display *display;
static wl_registry *registry;
static wl_shm *shm;
static wl_seat *seat;
static zwlr_screencopy_manager_v1 *manager;
static zwlr_virtual_pointer_manager_v1 *pointer_manager;
static zwp_virtual_keyboard_manager_v1 *keyboard_manager;
static zwlr_virtual_pointer_v1 *virtual_pointer;
static zwp_virtual_keyboard_v1 *virtual_keyboard;
static zwlr_foreign_toplevel_manager_v1 *toplevel_manager = nullptr;
static std::vector<wl_output *> outputs;
static std::vector<std::string> output_names;

static CaptureBackend detect_capture_backend() {
    // Check compositor type
    if (compositor_type == "kde" || compositor_type == "gnome") {
        if (pipewire_capture_available()) {
            std::cerr << "[CAPTURE] KDE/GNOME detected, using PipeWire\n";
            return PIPEWIRE;
        }
    }
    
    // Check if wlr-screencopy is available
    if (manager != nullptr) {
        std::cerr << "[CAPTURE] Using wlr-screencopy protocol\n";
        return WLR_SCREENCOPY;
    }
    
    // Fallback to PipeWire if available
    if (pipewire_capture_available()) {
        std::cerr << "[CAPTURE] Falling back to PipeWire\n";
        return PIPEWIRE;
    }
    
    std::cerr << "[CAPTURE] No capture backend available!\n";
    return WLR_SCREENCOPY; // Will fail later
}

// Focus tracking for output following
struct ToplevelInfo
{
    zwlr_foreign_toplevel_handle_v1 *handle;
    wl_output *current_output;
    bool activated;
};

struct FocusTracker
{
    std::atomic<int> focused_output_index{0};
    std::atomic<bool> follow_focus{false};
    std::mutex mutex;
    std::vector<ToplevelInfo> toplevels; // Store all toplevel windows
};

static FocusTracker focus_tracker;

struct Capture
{
    // Front buffer (being filled by compositor)
    wl_buffer *front_buffer = nullptr;
    void *front_data = nullptr;

    // Back buffer (stable, ready for rendering)
    std::vector<uint8_t> back_buffer;

    int width = 0, height = 0;
    int stride = 0;
    int size = 0;
    uint32_t format = 0;
    std::chrono::steady_clock::time_point timestamp;

    bool back_ready = false;
    bool front_ready = false;

    // Frame request state
    std::atomic<bool> frame_in_flight{false};

    // Default constructor
    Capture() = default;

    // Move constructor (needed for std::vector operations)
    Capture(Capture&& other) noexcept
        : front_buffer(other.front_buffer)
        , front_data(other.front_data)
        , back_buffer(std::move(other.back_buffer))
        , width(other.width)
        , height(other.height)
        , stride(other.stride)
        , size(other.size)
        , format(other.format)
        , timestamp(other.timestamp)
        , back_ready(other.back_ready)
        , front_ready(other.front_ready)
        , frame_in_flight(other.frame_in_flight.load())
    {
        other.front_buffer = nullptr;
        other.front_data = nullptr;
    }

    // Move assignment operator
    Capture& operator=(Capture&& other) noexcept {
        if (this != &other) {
            front_buffer = other.front_buffer;
            front_data = other.front_data;
            back_buffer = std::move(other.back_buffer);
            width = other.width;
            height = other.height;
            stride = other.stride;
            size = other.size;
            format = other.format;
            timestamp = other.timestamp;
            back_ready = other.back_ready;
            front_ready = other.front_ready;
            frame_in_flight.store(other.frame_in_flight.load());
            other.front_buffer = nullptr;
            other.front_data = nullptr;
        }
        return *this;
    }

    // Delete copy operations (atomic is not copyable)
    Capture(const Capture&) = delete;
    Capture& operator=(const Capture&) = delete;
};

// Frame cache per output
static std::vector<Capture> output_captures;
static std::vector<std::unique_ptr<std::mutex>> output_mutexes;
static std::vector<uint8_t> output_ready;
static std::atomic<bool> running{true};

static int frame_server_socket = -1;
static int input_server_socket = -1;

static std::atomic<bool> server_shift_pressed{false};
static std::atomic<bool> server_ctrl_pressed{false};
static std::atomic<bool> server_alt_pressed{false};
static std::atomic<bool> server_super_pressed{false};
static std::atomic<bool> server_altgr_pressed{false};
static std::atomic<bool> server_capslock_pressed{false};
static std::atomic<bool> server_numlock_pressed{false};

static bool feature_video = true;
static bool feature_audio = true;
static bool feature_input = true;
static bool feature_microphone = true;

// Input event queue
struct InputEvent
{
    MessageType type;
    std::string client_id;
    union
    {
        KeyEvent key;
        MouseMove mouse_move;
        MouseButton mouse_button;
        MouseScroll mouse_scroll;
    };
};

static std::queue<InputEvent> input_queue;
static std::mutex input_mutex;
static std::condition_variable input_cv;

// Host cursor position for zoom centering
static std::atomic<int> host_cursor_x{0};
static std::atomic<int> host_cursor_y{0};

static int create_shm_file(size_t size) {
  int fd = memfd_create("wayterm-mirror-shm", MFD_CLOEXEC);
  if (fd < 0) return -1;
  if (ftruncate(fd, size) < 0) {
    close(fd);
    return -1;
  }
  
  void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (addr != MAP_FAILED) {
    madvise(addr, size, MADV_SEQUENTIAL | MADV_WILLNEED);
    munmap(addr, size);
  }
  
  return fd;
}

// --- Frame listeners ---
struct CaptureContext
{
    int output_index;
    Capture *capture;
    std::mutex *mutex;
    uint8_t *ready;
};

static int detect_focused_output_hyprland()
{
    // Try hyprctl activewindow first (faster)
    std::string output = exec_command("hyprctl activewindow -j 2>/dev/null");
    if (output.empty())
        return -1;

    // Parse JSON to get monitor name
    rapidjson::Document doc;
    doc.Parse(output.c_str());
    if (doc.HasParseError() || !doc.IsObject())
        return -1;
    if (!doc.HasMember("monitor") || !doc["monitor"].IsInt64())
        return -1;
    int64_t monitor_name = doc["monitor"].GetInt64();
    return static_cast<int>(monitor_name);
}

static int detect_focused_output_sway()
{
    // Use swaymsg to get focused output
    std::string output = exec_command("swaymsg -t get_outputs -r 2>/dev/null");
    if (output.empty())
        return -1;

    // Parse JSON array
    rapidjson::Document doc;
    doc.Parse(output.c_str());
    if (doc.HasParseError() || !doc.IsArray())
        return -1;

    for (rapidjson::SizeType i = 0; i < doc.Size(); i++)
    {
        const rapidjson::Value &output_obj = doc[i];
        if (output_obj.HasMember("focused") && output_obj["focused"].IsBool() &&
            output_obj["focused"].GetBool() &&
            output_obj.HasMember("name") && output_obj["name"].IsString())
        {
            std::string monitor_name = output_obj["name"].GetString();
            // Try to match to our output list
            for (size_t j = 0; j < output_names.size(); j++)
            {
                if (output_names[j] == monitor_name)
                {
                    // std::cerr << "[FOCUS] Sway: Active window on " << monitor_name
                    //           << " (output " << j << ")\n";
                    return j;
                }
            }
        }
    }

    return -1;
}

static int detect_focused_output_kde()
{
    std::string output = exec_command("kscreen-doctor -o 2>/dev/null");
    if (output.empty())
        return -1;

    // std::cerr << "[FOCUS] KDE: CLI detection not fully implemented, using fallback\n";
    return -1;
}

static int detect_focused_output_gnome()
{
    // GNOME might use gdbus to query current monitor
    std::string output = exec_command(
        "gdbus call --session --dest org.gnome.Shell "
        "--object-path /org/gnome/Shell "
        "--method org.gnome.Shell.Eval 'global.display.focus_window.get_monitor()' 2>/dev/null");

    if (!output.empty())
    {
        // Try to parse monitor index from response
        size_t pos = output.find_first_of("0123456789");
        if (pos != std::string::npos)
        {
            int idx = output[pos] - '0';
            if (idx < (int)outputs.size())
            {
                std::cerr << "[FOCUS] GNOME: Active window on monitor " << idx << "\n";
                return idx;
            }
        }   
    }

    std::cerr << "[FOCUS] GNOME: CLI detection failed, using fallback\n";
    return -1;
}

// REPLACE detect_focused_output_index():
static int detect_focused_output_index()
{
    // Try compositor-specific CLI tools FIRST
    int result = -1;

    if (compositor_type == "hyprland")
    {
        result = detect_focused_output_hyprland();
    }
    else if (compositor_type == "sway")
    {
        result = detect_focused_output_sway();
    }
    else if (compositor_type == "kde")
    {
        result = detect_focused_output_kde();
    }
    else if (compositor_type == "gnome")
    {
        result = detect_focused_output_gnome();
    }

    // If CLI tool succeeded, return result
    if (result >= 0)
    {
        return result;
    }

    // FALLBACK: Use toplevel protocol (which is currently broken on many compositors)
    std::cerr << "[FOCUS] Using fallback toplevel protocol detection\n";

    std::lock_guard<std::mutex> lock(focus_tracker.mutex);

    // Find the activated window
    for (const auto &tl : focus_tracker.toplevels)
    {
        if (tl.activated && tl.current_output)
        {
            // Find which output index this is
            for (size_t i = 0; i < outputs.size(); i++)
            {
                if (outputs[i] == tl.current_output)
                {
                    std::cerr << "[FOCUS] Toplevel protocol: Focus on output " << i << "\n";
                    return i;
                }
            }
        }
    }

    // Last resort: return current value or 0
    int current = focus_tracker.focused_output_index.load();
    std::cerr << "[FOCUS] No focused window detected, using output " << current << "\n";
    return current;
}

static void update_focus_tracking()
{
    if (!focus_tracker.follow_focus.load())
        return;

    int new_output = detect_focused_output_index();
    int old_output = focus_tracker.focused_output_index.load();

    if (new_output != old_output)
    {
        focus_tracker.focused_output_index = new_output;
        std::cerr << "[FOCUS] *** Output switched from " << old_output
                  << " to " << new_output << " ***\n";
    }
}

static void focus_update_thread()
{
    std::cerr << "[FOCUS] Update thread started with compositor: " << compositor_type << "\n";

    while (running)
    {
        if (focus_tracker.follow_focus.load())
        {
            update_focus_tracking();
        }

        // Poll less frequently since we're using CLI tools (more expensive)
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::cerr << "[FOCUS] Update thread stopped\n";
}

static void frame_buffer(void *data, zwlr_screencopy_frame_v1 *frame,
                         uint32_t format, uint32_t width, uint32_t height, uint32_t stride)
{
    CaptureContext *ctx = static_cast<CaptureContext *>(data);
    std::lock_guard<std::mutex> lock(*ctx->mutex);

    Capture *cap = ctx->capture;

    // Clean up old buffer if exists
    if (cap->front_buffer)
    {
        munmap(cap->front_data, cap->size);
        wl_buffer_destroy(cap->front_buffer);
        cap->front_buffer = nullptr;
        cap->front_data = nullptr;
    }

    cap->width = width;
    cap->height = height;
    cap->stride = stride;
    cap->size = stride * height;
    cap->format = format;

    int fd = create_shm_file(cap->size);
    if (fd < 0)
        return;

    void *data_ptr = mmap(nullptr, cap->size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data_ptr == MAP_FAILED)
    {
        close(fd);
        return;
    }
    madvise(data_ptr, cap->size, MADV_SEQUENTIAL | MADV_WILLNEED);
    cap->front_data = data_ptr;
    wl_shm_pool *pool = wl_shm_create_pool(shm, fd, cap->size);
    cap->front_buffer = wl_shm_pool_create_buffer(pool, 0, width, height,
                                                  stride, format);
    wl_shm_pool_destroy(pool);
    close(fd);

    zwlr_screencopy_frame_v1_copy(frame, cap->front_buffer);
}

static void frame_flags(void *, zwlr_screencopy_frame_v1 *, uint32_t) {}

static void frame_ready_cb(void *data, zwlr_screencopy_frame_v1 *frame,
                           uint32_t, uint32_t, uint32_t)
{
    CaptureContext *ctx = static_cast<CaptureContext *>(data);
    std::lock_guard<std::mutex> lock(*ctx->mutex);

    Capture *cap = ctx->capture;

    // Mark front buffer ready
    cap->front_ready = true;
    cap->timestamp = std::chrono::steady_clock::now();
    cap->frame_in_flight = false;

    // Wake capture thread to swap buffers
    *ctx->ready = true;
    frame_ready_cvs[ctx->output_index]->notify_all();

    zwlr_screencopy_frame_v1_destroy(frame);
}

static void frame_failed(void *data, zwlr_screencopy_frame_v1 *frame)
{
    CaptureContext *ctx = static_cast<CaptureContext *>(data);
    std::lock_guard<std::mutex> lock(*ctx->mutex);

    *ctx->ready = false;
    zwlr_screencopy_frame_v1_destroy(frame);
}

static const zwlr_screencopy_frame_v1_listener frame_listener = {
    frame_buffer,
    frame_flags,
    frame_ready_cb,
    frame_failed};

// --- Registry ---
static void registry_add(void *, wl_registry *r, uint32_t name,
                         const char *iface, uint32_t version)
{
    if (strcmp(iface, wl_shm_interface.name) == 0)
        shm = static_cast<wl_shm *>(wl_registry_bind(r, name, &wl_shm_interface, 1));
    else if (strcmp(iface, wl_output_interface.name) == 0)
    {
        wl_output *out = static_cast<wl_output *>(wl_registry_bind(r, name, &wl_output_interface, 1));
        outputs.push_back(out);

        // Track output name for focus following
        std::ostringstream oss;
        oss << "output-" << (outputs.size() - 1);
        output_names.push_back(oss.str());

        std::cerr << "[OUTPUT] Registered " << output_names.back() << "\n";
    }
    else if (strcmp(iface, wl_seat_interface.name) == 0)
        seat = static_cast<wl_seat *>(wl_registry_bind(r, name, &wl_seat_interface, 1));
    else if (strcmp(iface, zwlr_screencopy_manager_v1_interface.name) == 0)
        manager = static_cast<zwlr_screencopy_manager_v1 *>(wl_registry_bind(r, name, &zwlr_screencopy_manager_v1_interface, 1));
    else if (strcmp(iface, zwlr_virtual_pointer_manager_v1_interface.name) == 0)
        pointer_manager = static_cast<zwlr_virtual_pointer_manager_v1 *>(wl_registry_bind(r, name, &zwlr_virtual_pointer_manager_v1_interface, std::min(version, 2u)));
    else if (strcmp(iface, zwp_virtual_keyboard_manager_v1_interface.name) == 0)
        keyboard_manager = static_cast<zwp_virtual_keyboard_manager_v1 *>(wl_registry_bind(r, name, &zwp_virtual_keyboard_manager_v1_interface, 1));
    else if (strcmp(iface, zwlr_foreign_toplevel_manager_v1_interface.name) == 0)
    {
        toplevel_manager = static_cast<zwlr_foreign_toplevel_manager_v1 *>(
            wl_registry_bind(r, name, &zwlr_foreign_toplevel_manager_v1_interface, 1));
        std::cerr << "[FOCUS] Foreign toplevel manager available\n";
    }
}

static void registry_remove(void *, wl_registry *, uint32_t) {}

static const wl_registry_listener registry_listener = {
    registry_add,
    registry_remove};
static void toplevel_handle_title(void *data, zwlr_foreign_toplevel_handle_v1 *handle, const char *title)
{
    // std::cerr << "[FOCUS] Window title: " << (title ? title : "(null)") << " handle=" << handle << "\n";
}

static void toplevel_handle_app_id(void *data, zwlr_foreign_toplevel_handle_v1 *handle, const char *app_id)
{
    // std::cerr << "[FOCUS] Window app_id: " << (app_id ? app_id : "(null)") << " handle=" << handle << "\n";
}

static void toplevel_handle_output_enter(void *data, zwlr_foreign_toplevel_handle_v1 *handle, wl_output *output)
{
    // std::cerr << "[FOCUS] Window " << handle << " entered output " << output << "\n";

    std::lock_guard<std::mutex> lock(focus_tracker.mutex);

    // Find or create toplevel info
    ToplevelInfo *info = nullptr;
    for (auto &tl : focus_tracker.toplevels)
    {
        if (tl.handle == handle)
        {
            info = &tl;
            break;
        }
    }

    if (!info)
    {
        focus_tracker.toplevels.push_back({handle, output, false});
        info = &focus_tracker.toplevels.back();
        std::cerr << "[FOCUS] Created new toplevel tracking for " << handle << "\n";
    }

    info->current_output = output;

    // Find which output index this is
    for (size_t i = 0; i < outputs.size(); i++)
    {
        if (outputs[i] == output)
        {
            // std::cerr << "[FOCUS] Window " << handle << " is on output " << i << "\n";

            // If this window is already activated (received state before output_enter),
            // update the focus tracker now
            if (info->activated)
            {
                focus_tracker.focused_output_index = i;
                // std::cerr << "[FOCUS] *** Switched to output " << i
                //           << " (late output_enter for already-activated window) ***\n";
            }
            break;
        }
    }
}

static void toplevel_handle_state(
    void *data,
    zwlr_foreign_toplevel_handle_v1 *handle,
    wl_array *state)
{
    bool activated = false;
    void *s;

    wl_array_for_each(s, state)
    {
        if (*(uint32_t *)s == ZWLR_FOREIGN_TOPLEVEL_HANDLE_V1_STATE_ACTIVATED)
        {
            activated = true;
            break;
        }
    }

    if (!activated)
        return;

    std::lock_guard<std::mutex> lock(focus_tracker.mutex);

    // Find the toplevel - it SHOULD already exist with output info
    for (auto &tl : focus_tracker.toplevels)
    {
        if (tl.handle == handle)
        {
            tl.activated = true;

            // We should already know the output from output_enter
            if (tl.current_output)
            {
                for (size_t i = 0; i < outputs.size(); i++)
                {
                    if (outputs[i] == tl.current_output)
                    {
                        focus_tracker.focused_output_index = i;
                        // std::cerr << "[FOCUS] *** Switched to output " << i << " (window already tracked) ***\n";
                        return;
                    }
                }
            }

            // Fallback: single output
            if (outputs.size() == 1)
            {
                focus_tracker.focused_output_index = 0;
                std::cerr << "[FOCUS] *** Switched to output 0 (single output fallback) ***\n";
            }
            else
            {
                // Window activated but we don't know its output yet
                // This happens for pre-existing windows during initialization
                // We'll get output_enter after this, so just log and wait
                // std::cerr << "[FOCUS] Window activated before output info received (handle="
                //          << handle << "), waiting for output_enter...\n";
            }
            return;
        }
    }

    // std::cerr << "[FOCUS] ERROR: Activated window not in tracking list! Handle=" << handle << "\n";
}

static void toplevel_handle_output_leave(void *data, zwlr_foreign_toplevel_handle_v1 *handle, wl_output *output)
{
    // std::cerr << "[FOCUS] Window " << handle << " left output " << output << "\n";

    std::lock_guard<std::mutex> lock(focus_tracker.mutex);

    for (auto &tl : focus_tracker.toplevels)
    {
        if (tl.handle == handle && tl.current_output == output)
        {
            tl.current_output = nullptr;
            break;
        }
    }
}

static void toplevel_handle_done(void *data, zwlr_foreign_toplevel_handle_v1 *handle)
{
    // std::cerr << "[FOCUS] Window " << handle << " done (batch complete)\n";
}

static void toplevel_handle_closed(void *data, zwlr_foreign_toplevel_handle_v1 *handle)
{
    // std::cerr << "[FOCUS] Window " << handle << " closed\n";

    std::lock_guard<std::mutex> lock(focus_tracker.mutex);

    // Remove from our tracking
    auto it = std::remove_if(focus_tracker.toplevels.begin(), focus_tracker.toplevels.end(),
                             [handle](const ToplevelInfo &tl)
                             { return tl.handle == handle; });

    focus_tracker.toplevels.erase(it, focus_tracker.toplevels.end());

    zwlr_foreign_toplevel_handle_v1_destroy(handle);
}

static void toplevel_handle_parent(void *, zwlr_foreign_toplevel_handle_v1 *, zwlr_foreign_toplevel_handle_v1 *) {}

static const zwlr_foreign_toplevel_handle_v1_listener toplevel_listener = {
    toplevel_handle_title,
    toplevel_handle_app_id,
    toplevel_handle_output_enter,
    toplevel_handle_output_leave,
    toplevel_handle_state,
    toplevel_handle_done,
    toplevel_handle_closed,
    toplevel_handle_parent,
};

static void manager_handle_toplevel(void *data, zwlr_foreign_toplevel_manager_v1 *mgr, zwlr_foreign_toplevel_handle_v1 *handle)
{
    // std::cerr << "[FOCUS] New toplevel discovered: " << handle << "\n";

    // Add listener BEFORE any roundtrip so we catch all events
    zwlr_foreign_toplevel_handle_v1_add_listener(handle, &toplevel_listener, nullptr);

    // Create entry immediately
    std::lock_guard<std::mutex> lock(focus_tracker.mutex);
    focus_tracker.toplevels.push_back({handle, nullptr, false});
    // std::cerr << "[FOCUS] Created tracking entry for " << handle << "\n";
}

static void manager_handle_finished(void *, zwlr_foreign_toplevel_manager_v1 *) {}

static const zwlr_foreign_toplevel_manager_v1_listener manager_listener = {
    manager_handle_toplevel,
    manager_handle_finished,
};

// --- Virtual input ---
static void setup_virtual_keyboard_keymap()
{
    if (!virtual_keyboard)
    {
        std::cerr << "Cannot setup keymap: virtual_keyboard is NULL\n";
        return;
    }

    std::string keymap_str =
        "xkb_keymap {\n"
        "  xkb_keycodes { include \"evdev+aliases(qwerty)\" };\n"
        "  xkb_types { include \"complete\" };\n"
        "  xkb_compat { include \"complete\" };\n"
        "  xkb_symbols { include \"pc+us+inet(evdev)\" };\n"
        "  xkb_geometry { include \"pc(pc105)\" };\n"
        "};\n";

    size_t keymap_size = keymap_str.size();
    int fd = create_shm_file(keymap_size);
    if (fd < 0)
    {
        std::cerr << "Failed to create keymap shm file\n";
        return;
    }

    void *data = mmap(nullptr, keymap_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED)
    {
        std::cerr << "Failed to mmap keymap\n";
        close(fd);
        return;
    }

    memcpy(data, keymap_str.c_str(), keymap_size);
    munmap(data, keymap_size);

    zwp_virtual_keyboard_v1_keymap(virtual_keyboard,
                                   WL_KEYBOARD_KEYMAP_FORMAT_XKB_V1, fd, keymap_size);

    close(fd);
    wl_display_flush(display);

    std::cerr << "Virtual keyboard keymap configured successfully\n";
}

static void handle_key_event(const KeyEvent &evt) {
    std::cerr << "[SERVER] Received key " << evt.keycode << " pressed=" << (int)evt.pressed
              << " [shift=" << (int)evt.shift << " ctrl=" << (int)evt.ctrl
              << " alt=" << (int)evt.alt << "]\n";

    uint32_t time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    bool is_shift_key = (evt.keycode == KEY_LEFTSHIFT || evt.keycode == KEY_RIGHTSHIFT);
    bool is_ctrl_key = (evt.keycode == KEY_LEFTCTRL || evt.keycode == KEY_RIGHTCTRL);
    bool is_alt_key = (evt.keycode == KEY_LEFTALT || evt.keycode == KEY_RIGHTALT);
    bool is_super_key = (evt.keycode == KEY_LEFTMETA || evt.keycode == KEY_RIGHTMETA);
    bool is_altgr_key = (evt.keycode == KEY_RIGHTALT);
    bool is_capslock_key = (evt.keycode == KEY_CAPSLOCK);
    bool is_numlock_key = (evt.keycode == KEY_NUMLOCK);

    if (is_shift_key) server_shift_pressed = evt.pressed;
    if (is_ctrl_key) server_ctrl_pressed = evt.pressed;
    if (is_alt_key) server_alt_pressed = evt.pressed;
    if (is_super_key) server_super_pressed = evt.pressed;
    if (is_altgr_key) server_altgr_pressed = evt.pressed;

    if (evt.pressed) {
        if (is_capslock_key) server_capslock_pressed = !server_capslock_pressed;
        if (is_numlock_key) server_numlock_pressed = !server_numlock_pressed;
    }

    // Try virtual input manager first, fallback to Wayland protocols
    if (virtual_input_mgr.backend == VirtualInputManager::UINPUT) {
        virtual_input_mgr.send_key(evt.keycode, evt.pressed);
    } else if (virtual_keyboard) {
        uint32_t state = evt.pressed ? WL_KEYBOARD_KEY_STATE_PRESSED : WL_KEYBOARD_KEY_STATE_RELEASED;
        zwp_virtual_keyboard_v1_key(virtual_keyboard, time_ms, evt.keycode, state);

        uint32_t mods_depressed = 0;
        uint32_t mods_locked = 0;

        if (server_shift_pressed) mods_depressed |= (1 << 0);
        if (server_ctrl_pressed) mods_depressed |= (1 << 2);
        if (server_alt_pressed) mods_depressed |= (1 << 3);
        if (server_super_pressed) mods_depressed |= (1 << 6);
        if (server_altgr_pressed) mods_depressed |= (1 << 7);

        if (server_capslock_pressed) mods_locked |= (1 << 1);
        if (server_numlock_pressed) mods_locked |= (1 << 4);

        zwp_virtual_keyboard_v1_modifiers(virtual_keyboard,
                                          mods_depressed, 0, mods_locked, 0);

        wl_display_flush(display);
    }

    std::cerr << "[SERVER] Key event sent\n";
}

static void handle_mouse_move(const MouseMove &evt, const std::string &client_id) {
    std::shared_ptr<ClientConnection> conn;
    uint32_t output_index = 0;
    uint32_t output_width = 0;
    uint32_t output_height = 0;

    {
        std::lock_guard<std::mutex> lock(clients_mutex);
        auto it = clients.find(client_id);
        if (it != clients.end()) {
            conn = it->second;
            output_index = conn->config.follow_focus
                ? focus_tracker.focused_output_index.load()
                : conn->config.output_index;
            output_index = std::min(output_index, (uint32_t)(outputs.size() - 1));

            std::lock_guard<std::mutex> output_lock(*output_mutexes[output_index]);
            Capture &cap = output_captures[output_index];
            if (cap.width > 0 && cap.height > 0) {
                output_width = cap.width;
                output_height = cap.height;
            }
        }
    }

    if (output_width == 0 || output_height == 0) {
        output_width = evt.width;
        output_height = evt.height;
    }

    host_cursor_x = evt.x;
    host_cursor_y = evt.y;

    // Try virtual input manager first
    if (virtual_input_mgr.backend == VirtualInputManager::UINPUT) {
        virtual_input_mgr.send_mouse_move(evt.x, evt.y, output_width, output_height);
    } else if (virtual_pointer) {
        uint32_t time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        zwlr_virtual_pointer_v1_motion_absolute(virtual_pointer,
                                                time_ms,
                                                (uint32_t)evt.x,
                                                (uint32_t)evt.y,
                                                (uint32_t)output_width,
                                                (uint32_t)output_height);

        zwlr_virtual_pointer_v1_frame(virtual_pointer);
        wl_display_flush(display);
    }
}

static void handle_mouse_button(const MouseButton &evt) {
    std::cerr << "[SERVER] Mouse button: button=" << evt.button
              << " pressed=" << (int)evt.pressed << "\n";

    uint32_t linux_button = BTN_LEFT;
    if (evt.button == 2) linux_button = BTN_MIDDLE;
    else if (evt.button == 3) linux_button = BTN_RIGHT;

    // Try virtual input manager first
    if (virtual_input_mgr.backend == VirtualInputManager::UINPUT) {
        virtual_input_mgr.send_mouse_button(linux_button, evt.pressed);
    } else if (virtual_pointer) {
        uint32_t time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        zwlr_virtual_pointer_v1_button(virtual_pointer, time_ms, linux_button,
                                       evt.pressed ? WL_POINTER_BUTTON_STATE_PRESSED 
                                                   : WL_POINTER_BUTTON_STATE_RELEASED);

        zwlr_virtual_pointer_v1_frame(virtual_pointer);
        wl_display_flush(display);
    }

    std::cerr << "[SERVER] Mouse button sent\n";
}

static void handle_mouse_scroll(const MouseScroll &evt) {
    std::cerr << "[SERVER] Mouse scroll: direction=" << evt.direction << "\n";

    // Try virtual input manager first
    if (virtual_input_mgr.backend == VirtualInputManager::UINPUT) {
        virtual_input_mgr.send_mouse_scroll(evt.direction);
    } else if (virtual_pointer) {
        uint32_t time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        wl_fixed_t value = wl_fixed_from_int(evt.direction * 15);

        zwlr_virtual_pointer_v1_axis(virtual_pointer,
                                     time_ms, WL_POINTER_AXIS_VERTICAL_SCROLL, value);

        zwlr_virtual_pointer_v1_frame(virtual_pointer);
        wl_display_flush(display);
    }

    std::cerr << "[SERVER] Mouse scroll sent\n";
}

// --- Input processing thread ---
static void input_thread()
{
    while (running)
    {
        std::unique_lock<std::mutex> lock(input_mutex);
        input_cv.wait_for(lock, std::chrono::milliseconds(1), // CHANGED: 1ms instead of 10ms
                          []
                          { return !input_queue.empty() || !running; });

        if (!running)
            break;

        while (!input_queue.empty())
        {
            InputEvent evt = input_queue.front();
            input_queue.pop();
            lock.unlock();

            switch (evt.type)
            {
            case MessageType::KEY_EVENT:
                handle_key_event(evt.key);
                break;
            case MessageType::MOUSE_MOVE:
                handle_mouse_move(evt.mouse_move, evt.client_id);
                break;
            case MessageType::MOUSE_BUTTON:
                handle_mouse_button(evt.mouse_button);
                break;
            case MessageType::MOUSE_SCROLL:
                handle_mouse_scroll(evt.mouse_scroll);
                break;
            default:
                break;
            }

            lock.lock();
        }
    }
}

// ============================================================================
// COMPLETE RENDERER REWRITE - PROPER IMPLEMENTATION
// ============================================================================

// Color distance using proper perceptual weighting
static double color_distance(uint8_t r1, uint8_t g1, uint8_t b1,
                             uint8_t r2, uint8_t g2, uint8_t b2)
{
    // CIE76 approximation - green most important for human eye
    double dr = r1 - r2;
    double dg = g1 - g2;
    double db = b1 - b2;
    return sqrt(2.0 * dr * dr + 4.0 * dg * dg + 3.0 * db * db);
}

// ANSI 16 color mapping
static uint8_t rgb_to_ansi_16(uint8_t r, uint8_t g, uint8_t b)
{
    uint8_t intensity = (r > 128 || g > 128 || b > 128) ? 8 : 0;
    uint8_t code = intensity;
    if (r > 64)
        code |= 1;
    if (g > 64)
        code |= 2;
    if (b > 64)
        code |= 4;
    return code;
}

static uint8_t rgb_to_ansi_256(uint8_t r, uint8_t g, uint8_t b)
{
    // Calculate grayscale value
    uint8_t gray = (r * 30 + g * 59 + b * 11) / 100;

    // Check if color is truly grayscale (very low saturation)
    uint8_t min_rgb = std::min({r, g, b});
    uint8_t max_rgb = std::max({r, g, b});
    uint8_t saturation = max_rgb - min_rgb;

    // FIXED: Much stricter grayscale detection (saturation < 3 instead of < 12)
    // Most "gray-ish" colors should still use RGB cube
    if (saturation < 3)
    {
        if (gray < 8)
            return 16; // Black
        if (gray > 247)
            return 231; // White
        // 24 gray shades (232-255)
        int gray_idx = (gray - 8) * 24 / 239;
        return 232 + std::clamp(gray_idx, 0, 23);
    }

    // FIXED: Better RGB cube mapping with proper rounding
    // 6x6x6 RGB cube (16-231)
    int r_idx = std::clamp((r * 6 + 128) / 256, 0, 5);
    int g_idx = std::clamp((g * 6 + 128) / 256, 0, 5);
    int b_idx = std::clamp((b * 6 + 128) / 256, 0, 5);

    return 16 + (r_idx * 36) + (g_idx * 6) + b_idx;
}

// Foreground color code
static std::string rgb_to_ansi(uint8_t r, uint8_t g, uint8_t b, ColorMode mode)
{
    switch (mode)
    {
    case ColorMode::ANSI_16:
        return "\033[38;5;" + std::to_string(rgb_to_ansi_16(r, g, b)) + "m";
    case ColorMode::ANSI_256:
        return "\033[38;5;" + std::to_string(rgb_to_ansi_256(r, g, b)) + "m";
    case ColorMode::ANSI_TRUECOLOR:
        return "\033[38;2;" + std::to_string((int)r) + ";" + std::to_string((int)g) + ";" + std::to_string((int)b) + "m";
    default:
        return "";
    }
}

// Background color code
static std::string rgb_to_ansi_bg(uint8_t r, uint8_t g, uint8_t b, ColorMode mode)
{
    switch (mode)
    {
    case ColorMode::ANSI_16:
        return "\033[48;5;" + std::to_string(rgb_to_ansi_16(r, g, b)) + "m";
    case ColorMode::ANSI_256:
        return "\033[48;5;" + std::to_string(rgb_to_ansi_256(r, g, b)) + "m";
    case ColorMode::ANSI_TRUECOLOR:
        return "\033[48;2;" + std::to_string((int)r) + ";" + std::to_string((int)g) + ";" + std::to_string((int)b) + "m";
    default:
        return "";
    }
}

// Extract RGB from BGRA frame
static inline void get_rgb(const uint8_t *p, uint8_t &r, uint8_t &g, uint8_t &b)
{
    b = p[0];
    g = p[1];
    r = p[2];
}

// Helper to sample a pixel with rotation transformation
// rotation_angle is in degrees (0-360, any value)
static inline void sample_rotated_pixel(
    const uint8_t *frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int x, int y,
    int rotated_width, int rotated_height,
    double rotation_angle,
    uint8_t &r, uint8_t &g, uint8_t &b)
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
    
    const uint8_t *p = frame_data + src_y * frame_stride + src_x * 4;
    get_rgb(p, r, g, b);
}

// Get effective dimensions after rotation (bounding box)
static inline void get_rotated_dimensions(
    uint32_t frame_width, uint32_t frame_height,
    double rotation_angle,
    uint32_t &out_width, uint32_t &out_height)
{
    double rad = fabs(rotation_angle) * M_PI / 180.0;
    double cos_a = fabs(cos(rad));
    double sin_a = fabs(sin(rad));
    
    // Bounding box of rotated rectangle
    out_width = (uint32_t)ceil(frame_width * cos_a + frame_height * sin_a);
    out_height = (uint32_t)ceil(frame_width * sin_a + frame_height * cos_a);
}

struct RegionalColorAnalysis
{
    uint8_t dominant_fg_r, dominant_fg_g, dominant_fg_b;
    uint8_t dominant_bg_r, dominant_bg_g, dominant_bg_b;
    double contrast;
};

static RegionalColorAnalysis analyze_regional_colors(
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
    uint8_t detail_level)
{

    // Sample a larger region around this cell
    int region_size = 5; // 5×5 cells = 10×20 pixels
    if (detail_level >= 90)
    {
        region_size = 3; // Smaller region for high detail
    }
    else if (detail_level < 50)
    {
        region_size = 7; // Larger region for smooth colors
    }

    // Collect all pixel colors in the region
    std::vector<uint8_t> reds, greens, blues, lumas;
    reds.reserve(region_size * region_size * 8);
    greens.reserve(region_size * region_size * 8);
    blues.reserve(region_size * region_size * 8);
    lumas.reserve(region_size * region_size * 8);

    int half_region = region_size / 2;

    for (int dy = -half_region; dy <= half_region; dy++)
    {
        for (int dx = -half_region; dx <= half_region; dx++)
        {
            int nx = cell_x + dx;
            int ny = cell_y + dy;

            // Clamp to valid cells
            if (nx < 0 || nx >= cells_x || ny < 0 || ny >= cells_y)
                continue;

            // Sample all 8 dots in this neighboring cell
            for (int dot = 0; dot < 8; dot++)
            {
                int dot_positions[8][2] = {
                    {0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {0, 3}, {1, 3}};

                int dot_x = dot_positions[dot][0];
                int dot_y = dot_positions[dot][1];

                int src_x = (nx * 2 + dot_x) * frame_width / w_scaled;
                int src_y = (ny * 4 + dot_y) * frame_height / h_scaled;

                src_x = std::clamp(src_x, 0, (int)frame_width - 1);
                src_y = std::clamp(src_y, 0, (int)frame_height - 1);

                const uint8_t *p = frame_data + src_y * frame_stride + src_x * 4;
                uint8_t r, g, b;
                get_rgb(p, r, g, b);

                reds.push_back(r);
                greens.push_back(g);
                blues.push_back(b);
                lumas.push_back(0.299 * r + 0.587 * g + 0.114 * b);
            }
        }
    }

    if (reds.empty())
    {
        // Fallback
        return {128, 128, 128, 64, 64, 64, 64.0};
    }

    // Find median luminance to split into light/dark groups
    std::vector<uint8_t> sorted_lumas = lumas;
    std::sort(sorted_lumas.begin(), sorted_lumas.end());
    uint8_t median_luma = sorted_lumas[sorted_lumas.size() / 2];

    // Calculate mean colors for light pixels (foreground) and dark pixels (background)
    uint32_t fg_r_sum = 0, fg_g_sum = 0, fg_b_sum = 0, fg_count = 0;
    uint32_t bg_r_sum = 0, bg_g_sum = 0, bg_b_sum = 0, bg_count = 0;

    for (size_t i = 0; i < lumas.size(); i++)
    {
        if (lumas[i] >= median_luma)
        {
            // Light pixel (foreground)
            fg_r_sum += reds[i];
            fg_g_sum += greens[i];
            fg_b_sum += blues[i];
            fg_count++;
        }
        else
        {
            // Dark pixel (background)
            bg_r_sum += reds[i];
            bg_g_sum += greens[i];
            bg_b_sum += blues[i];
            bg_count++;
        }
    }

    RegionalColorAnalysis result;

    if (fg_count > 0)
    {
        result.dominant_fg_r = fg_r_sum / fg_count;
        result.dominant_fg_g = fg_g_sum / fg_count;
        result.dominant_fg_b = fg_b_sum / fg_count;
    }
    else
    {
        result.dominant_fg_r = result.dominant_fg_g = result.dominant_fg_b = 200;
    }

    if (bg_count > 0)
    {
        result.dominant_bg_r = bg_r_sum / bg_count;
        result.dominant_bg_g = bg_g_sum / bg_count;
        result.dominant_bg_b = bg_b_sum / bg_count;
    }
    else
    {
        result.dominant_bg_r = result.dominant_bg_g = result.dominant_bg_b = 55;
    }

    // Calculate perceptual contrast
    double fg_luma = 0.299 * result.dominant_fg_r + 0.587 * result.dominant_fg_g + 0.114 * result.dominant_fg_b;
    double bg_luma = 0.299 * result.dominant_bg_r + 0.587 * result.dominant_bg_g + 0.114 * result.dominant_bg_b;
    result.contrast = std::abs(fg_luma - bg_luma);

    return result;
}

static BrailleCell analyze_braille_cell(
    const uint8_t *frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int cell_x,
    int cell_y,
    int w_scaled,
    int h_scaled,
    uint8_t detail_level)
{

    BrailleCell cell;
    memset(&cell, 0, sizeof(cell));

    // Calculate sampling kernel size based on detail level
    int kernel_size;
    if (detail_level >= 95)
    {
        kernel_size = 1;
    }
    else if (detail_level >= 80)
    {
        kernel_size = 2;
    }
    else if (detail_level >= 60)
    {
        kernel_size = 3;
    }
    else if (detail_level >= 40)
    {
        kernel_size = 4;
    }
    else
    {
        kernel_size = 2; // Low detail: just use 2x2 sampling, not 5x5 (WAY faster)
    }

    // Braille dot positions (2x4 grid)
    int dot_positions[8][2] = {
        {0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {0, 3}, {1, 3}};

    // Sample each dot
    for (int dot = 0; dot < 8; dot++)
    {
        int dot_x = dot_positions[dot][0];
        int dot_y = dot_positions[dot][1];

        int src_x = (cell_x * 2 + dot_x) * frame_width / w_scaled;
        int src_y = (cell_y * 4 + dot_y) * frame_height / h_scaled;

        // Sample with kernel
        uint32_t r_sum = 0, g_sum = 0, b_sum = 0;
        double weight_sum = 0;

        for (int ky = 0; ky < kernel_size; ky++)
        {
            for (int kx = 0; kx < kernel_size; kx++)
            {
                int px = std::clamp(src_x + kx - kernel_size / 2, 0, (int)frame_width - 1);
                int py = std::clamp(src_y + ky - kernel_size / 2, 0, (int)frame_height - 1);

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

        cell.colors[dot][0] = r;
        cell.colors[dot][1] = g;
        cell.colors[dot][2] = b;

        cell.lumas[dot] = 0.299 * r + 0.587 * g + 0.114 * b;
    }

    // CRITICAL SIMPLE EDGE DETECTION: Does ANY edge exist?
    // Check all adjacent pairs - if ANY pair has significant difference, edge exists

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

    cell.has_edge = false;
    double max_edge = 0;

    // Edge detection threshold - adjust based on detail level
    double edge_threshold = (detail_level >= 70) ? 30.0 : 50.0;

    for (auto &pair : adjacency_pairs)
    {
        int i = pair[0];
        int j = pair[1];

        // Calculate perceptual color distance
        double dr = cell.colors[i][0] - cell.colors[j][0];
        double dg = cell.colors[i][1] - cell.colors[j][1];
        double db = cell.colors[i][2] - cell.colors[j][2];
        double color_dist = sqrt(2.0 * dr * dr + 4.0 * dg * dg + 3.0 * db * db);

        // Luminance difference
        double luma_diff = std::abs(cell.lumas[i] - cell.lumas[j]);

        // Edge strength (use max of color or luma difference)
        double edge = std::max(color_dist, luma_diff * 2.0);

        max_edge = std::max(max_edge, edge);

        // If ANY edge exceeds threshold, we have an edge
        if (edge > edge_threshold)
        {
            cell.has_edge = true;
            // Don't break - we still want max_edge for threshold calc
        }
    }

    cell.edge_strength = max_edge;

    // Calculate mean luma
    double sum = 0;
    for (int i = 0; i < 8; i++)
    {
        sum += cell.lumas[i];
    }
    cell.mean_luma = sum / 8.0;

    // Weight calculation
    for (int i = 0; i < 8; i++)
    {
        double contrast = std::abs(cell.lumas[i] - cell.mean_luma);
        cell.weights[i] = 1.0 + (contrast / 128.0);
    }

    return cell;
}

static void boost_braille_contrast(
    uint8_t &fg_r, uint8_t &fg_g, uint8_t &fg_b,
    uint8_t &bg_r, uint8_t &bg_g, uint8_t &bg_b,
    double contrast_target)
{

    double fg_luma = 0.299 * fg_r + 0.587 * fg_g + 0.114 * fg_b;
    double bg_luma = 0.299 * bg_r + 0.587 * bg_g + 0.114 * bg_b;
    double current_contrast = std::abs(fg_luma - bg_luma);

    if (current_contrast >= contrast_target)
        return;

    double boost_factor = contrast_target / (current_contrast + 1e-5);

    if (fg_luma > bg_luma)
    {
        // Make foreground lighter, background darker
        fg_r = std::min(255, (int)(fg_r * boost_factor));
        fg_g = std::min(255, (int)(fg_g * boost_factor));
        fg_b = std::min(255, (int)(fg_b * boost_factor));

        bg_r = std::max(0, (int)(bg_r / boost_factor));
        bg_g = std::max(0, (int)(bg_g / boost_factor));
        bg_b = std::max(0, (int)(bg_b / boost_factor));
    }
    else
    {
        // Make background lighter, foreground darker
        bg_r = std::min(255, (int)(bg_r * boost_factor));
        bg_g = std::min(255, (int)(bg_g * boost_factor));
        bg_b = std::min(255, (int)(bg_b * boost_factor));

        fg_r = std::max(0, (int)(fg_r / boost_factor));
        fg_g = std::max(0, (int)(fg_g / boost_factor));
        fg_b = std::max(0, (int)(fg_b / boost_factor));
    }
}

// FIXED: Braille pattern calculation - proper inversion based on what's majority
static uint8_t calculate_braille_pattern(
    const BrailleCell &cell,
    uint8_t detail_level,
    uint8_t quality,
    const RegionalColorAnalysis &regional)
{

    int step = std::max(1, 16 - (quality / 7));

    // HIGH DETAIL (>= 90): Otsu on luminance
    if (detail_level >= 90)
    {
        double best_threshold = cell.mean_luma;
        double best_separation = 0;

        for (int t_int = 0; t_int <= 255; t_int += step)
        {
            double t = t_int;
            double sum_below = 0, sum_above = 0;
            int count_below = 0, count_above = 0;

            for (int i = 0; i < 8; i++)
            {
                if (cell.lumas[i] < t)
                {
                    sum_below += cell.lumas[i];
                    count_below++;
                }
                else
                {
                    sum_above += cell.lumas[i];
                    count_above++;
                }
            }

            if (count_below > 0 && count_above > 0)
            {
                double mean_below = sum_below / count_below;
                double mean_above = sum_above / count_above;
                double separation = count_below * count_above * (mean_above - mean_below) * (mean_above - mean_below);

                if (separation > best_separation)
                {
                    best_separation = separation;
                    best_threshold = t;
                }
            }
        }

        // Apply threshold
        uint8_t pattern = 0;

        for (int dot = 0; dot < 8; dot++)
        {
            if (cell.lumas[dot] > best_threshold)
            {
                pattern |= (1 << dot);
            }
        }

        // FIXED: Invert if MAJORITY of dots are lit (means we're drawing background)
        int lit_count = __builtin_popcount(pattern);
        if (lit_count > 4)
        {
            pattern = ~pattern;
        }

        return pattern;
    }

    // MEDIUM-HIGH DETAIL (70-89): Otsu on luminance
    double threshold;
    if (detail_level >= 70)
    {
        double best_threshold = cell.mean_luma;
        double best_separation = 0;

        for (int t_int = 0; t_int <= 255; t_int += step)
        {
            double t = t_int;
            double sum_below = 0, sum_above = 0;
            int count_below = 0, count_above = 0;

            for (int i = 0; i < 8; i++)
            {
                if (cell.lumas[i] < t)
                {
                    sum_below += cell.lumas[i];
                    count_below++;
                }
                else
                {
                    sum_above += cell.lumas[i];
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
    }
    else if (detail_level >= 40)
    {
        threshold = cell.mean_luma;
    }
    else
    {
        threshold = cell.mean_luma;
    }

    threshold = std::clamp(threshold, 0.0, 255.0);

    // Apply threshold
    uint8_t pattern = 0;

    for (int dot = 0; dot < 8; dot++)
    {
        if (cell.lumas[dot] > threshold)
        {
            pattern |= (1 << dot);
        }
    }

    // FIXED: Invert if MAJORITY of dots are lit (means we're drawing background)
    int lit_count = __builtin_popcount(pattern);
    if (lit_count > 4)
    {
        pattern = ~pattern;
    }

    return pattern;
}

// Color assignment - no changes needed
static void calculate_braille_colors(
    const BrailleCell &cell,
    uint8_t pattern,
    uint8_t &fg_r, uint8_t &fg_g, uint8_t &fg_b,
    uint8_t &bg_r, uint8_t &bg_g, uint8_t &bg_b,
    uint8_t detail_level,
    const RegionalColorAnalysis &regional)
{

    std::vector<int> lit_dots, unlit_dots;

    for (int i = 0; i < 8; i++)
    {
        if (pattern & (1 << i))
        {
            lit_dots.push_back(i);
        }
        else
        {
            unlit_dots.push_back(i);
        }
    }

    // Calculate average colors
    uint32_t lit_r = 0, lit_g = 0, lit_b = 0;
    uint32_t unlit_r = 0, unlit_g = 0, unlit_b = 0;

    for (int dot : lit_dots)
    {
        lit_r += cell.colors[dot][0];
        lit_g += cell.colors[dot][1];
        lit_b += cell.colors[dot][2];
    }
    for (int dot : unlit_dots)
    {
        unlit_r += cell.colors[dot][0];
        unlit_g += cell.colors[dot][1];
        unlit_b += cell.colors[dot][2];
    }

    if (lit_dots.empty())
    {
        fg_r = regional.dominant_bg_r;
        fg_g = regional.dominant_bg_g;
        fg_b = regional.dominant_bg_b;
        bg_r = fg_r;
        bg_g = fg_g;
        bg_b = fg_b;
        return;
    }
    else if (unlit_dots.empty())
    {
        fg_r = regional.dominant_fg_r;
        fg_g = regional.dominant_fg_g;
        fg_b = regional.dominant_fg_b;
        bg_r = fg_r;
        bg_g = fg_g;
        bg_b = fg_b;
        return;
    }

    // After inversion (if applied), lit dots = content, unlit = background
    fg_r = lit_r / lit_dots.size();
    fg_g = lit_g / lit_dots.size();
    fg_b = lit_b / lit_dots.size();

    bg_r = unlit_r / unlit_dots.size();
    bg_g = unlit_g / unlit_dots.size();
    bg_b = unlit_b / unlit_dots.size();
}

// Render with hybrid braille and half-block (high detail braille where needed)

static std::string render_braille(
    const uint8_t *frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int term_width,
    int term_height,
    ColorMode mode,
    bool keep_aspect_ratio,
    double scale_factor,
    uint8_t detail_level,
    uint8_t threshold_steps,
    double rotation_angle)
{

    if (!frame_data || frame_width == 0 || frame_height == 0)
        return "";

    std::ostringstream out;
    out.precision(0);
    out << std::fixed;

    // Get rotated dimensions
    uint32_t rot_width, rot_height;
    get_rotated_dimensions(frame_width, frame_height, rotation_angle, rot_width, rot_height);

    // Calculate scaled dimensions (2x4 pixels per cell) using rotated dimensions
    int w_scaled, h_scaled;
    if (keep_aspect_ratio)
    {
        double src_aspect = (double)rot_width / rot_height;
        double term_aspect = (double)(term_width * 2) / (term_height * 4);

        if (src_aspect > term_aspect)
        {
            w_scaled = term_width * 2;
            h_scaled = (int)(w_scaled / src_aspect);
        }
        else
        {
            h_scaled = term_height * 4;
            w_scaled = (int)(h_scaled * src_aspect);
        }
    }
    else
    {
        w_scaled = (int)(rot_width * scale_factor);
        h_scaled = (int)(rot_height * scale_factor);
    }

    w_scaled = std::clamp(w_scaled, 2, term_width * 2);
    h_scaled = std::clamp(h_scaled, 4, term_height * 4);

    int cells_x = w_scaled / 2;
    int cells_y = h_scaled / 4;

    // Calculate kernel size based on detail
    int kernel_size;
    if (detail_level >= 95) kernel_size = 1;
    else if (detail_level >= 80) kernel_size = 2;
    else if (detail_level >= 60) kernel_size = 3;
    else kernel_size = 2;

    // Braille dot positions (2x4 grid)
    int dot_positions[8][2] = {
        {0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {0, 3}, {1, 3}};

    // Render each braille cell
    for (int cy = 0; cy < cells_y; cy++)
    {
        for (int cx = 0; cx < cells_x; cx++)
        {
            // Sample all 8 dots with rotation
            double lumas[8];
            uint8_t colors[8][3];

            for (int dot = 0; dot < 8; dot++)
            {
                int dot_x = dot_positions[dot][0];
                int dot_y = dot_positions[dot][1];

                int rot_x = (cx * 2 + dot_x) * rot_width / w_scaled;
                int rot_y = (cy * 4 + dot_y) * rot_height / h_scaled;

                // Sample with kernel
                uint32_t r_sum = 0, g_sum = 0, b_sum = 0;
                double weight_sum = 0;

                for (int ky = 0; ky < kernel_size; ky++)
                {
                    for (int kx = 0; kx < kernel_size; kx++)
                    {
                        int px = std::clamp(rot_x + kx - kernel_size / 2, 0, (int)rot_width - 1);
                        int py = std::clamp(rot_y + ky - kernel_size / 2, 0, (int)rot_height - 1);

                        uint8_t r, g, b;
                        sample_rotated_pixel(frame_data, frame_width, frame_height, frame_stride,
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

                colors[dot][0] = r_sum / weight_sum;
                colors[dot][1] = g_sum / weight_sum;
                colors[dot][2] = b_sum / weight_sum;
                lumas[dot] = 0.299 * colors[dot][0] + 0.587 * colors[dot][1] + 0.114 * colors[dot][2];
            }

            // Calculate mean luma for threshold
            double mean_luma = 0;
            for (int i = 0; i < 8; i++) mean_luma += lumas[i];
            mean_luma /= 8.0;

            // Calculate pattern using simple threshold
            uint8_t pattern = 0;
            for (int dot = 0; dot < 8; dot++)
            {
                if (lumas[dot] > mean_luma)
                    pattern |= (1 << dot);
            }

            // Invert if majority lit
            int lit_count = __builtin_popcount(pattern);
            if (lit_count > 4)
                pattern = ~pattern;

            // Calculate colors from lit/unlit dots
            uint32_t fg_r = 0, fg_g = 0, fg_b = 0, fg_count = 0;
            uint32_t bg_r = 0, bg_g = 0, bg_b = 0, bg_count = 0;

            for (int i = 0; i < 8; i++)
            {
                if (pattern & (1 << i))
                {
                    fg_r += colors[i][0];
                    fg_g += colors[i][1];
                    fg_b += colors[i][2];
                    fg_count++;
                }
                else
                {
                    bg_r += colors[i][0];
                    bg_g += colors[i][1];
                    bg_b += colors[i][2];
                    bg_count++;
                }
            }

            uint8_t fr = fg_count ? fg_r / fg_count : 128;
            uint8_t fg = fg_count ? fg_g / fg_count : 128;
            uint8_t fb = fg_count ? fg_b / fg_count : 128;
            uint8_t br = bg_count ? bg_r / bg_count : 64;
            uint8_t bgc = bg_count ? bg_g / bg_count : 64;
            uint8_t bb = bg_count ? bg_b / bg_count : 64;

            // Output ANSI codes and braille character
            out << rgb_to_ansi(fr, fg, fb, mode);
            out << rgb_to_ansi_bg(br, bgc, bb, mode);

            // Convert pattern to Unicode braille character
            int codepoint = 0x2800 + pattern;
            out << (char)(0xE0 | (codepoint >> 12))
                << (char)(0x80 | ((codepoint >> 6) & 0x3F))
                << (char)(0x80 | (codepoint & 0x3F));
        }

        out << "\033[0m";
        if (cy < cells_y - 1)
            out << '\n';
    }

    return out.str();
}

// CUDA-accelerated BRAILLE renderer (2x4 dots, high detail)
// Now supports rotation natively in CUDA
static std::string render_braille_cuda_wrapper(
    const uint8_t *frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int term_width,
    int term_height,
    ColorMode mode,
    bool keep_aspect_ratio,
    double scale_factor,
    uint8_t detail_level,
    uint8_t threshold_steps,
    double rotation_angle)
{
    if (!frame_data || frame_width == 0 || frame_height == 0)
        return "";

    std::ostringstream out;
    out.precision(0);
    out << std::fixed;

    // Get rotated dimensions for aspect ratio calculation
    uint32_t rot_width, rot_height;
    get_rotated_dimensions(frame_width, frame_height, rotation_angle, rot_width, rot_height);

    // Calculate scaled dimensions based on rotated image size
    int w_scaled, h_scaled;
    if (keep_aspect_ratio)
    {
        double src_aspect = (double)rot_width / rot_height;
        double term_aspect = (double)(term_width * 2) / (term_height * 4);

        if (src_aspect > term_aspect)
        {
            w_scaled = term_width * 2;
            h_scaled = (int)(w_scaled / src_aspect);
        }
        else
        {
            h_scaled = term_height * 4;
            w_scaled = (int)(h_scaled * src_aspect);
        }
    }
    else
    {
        w_scaled = (int)(rot_width * scale_factor);
        h_scaled = (int)(rot_height * scale_factor);
    }

    w_scaled = std::clamp(w_scaled, 2, term_width * 2);
    h_scaled = std::clamp(h_scaled, 4, term_height * 4);

    int cells_x = w_scaled / 2;
    int cells_y = h_scaled / 4;

    // Allocate output buffers
    std::vector<uint8_t> patterns(cells_x * cells_y);
    std::vector<uint8_t> fg_colors(cells_x * cells_y * 3);
    std::vector<uint8_t> bg_colors(cells_x * cells_y * 3);

    // Call CUDA kernel with rotation support
    render_braille_cuda(
        frame_data, frame_width, frame_height, frame_stride,
        w_scaled, h_scaled, cells_x, cells_y,
        rotation_angle,
        detail_level, threshold_steps,
        patterns.data(), fg_colors.data(), bg_colors.data());

    // Convert results to ANSI string
    for (int cy = 0; cy < cells_y; cy++)
    {
        for (int cx = 0; cx < cells_x; cx++)
        {
            int idx = cy * cells_x + cx;
            uint8_t pattern = patterns[idx];

            uint8_t fg_r = fg_colors[idx * 3 + 0];
            uint8_t fg_g = fg_colors[idx * 3 + 1];
            uint8_t fg_b = fg_colors[idx * 3 + 2];

            uint8_t bg_r = bg_colors[idx * 3 + 0];
            uint8_t bg_g = bg_colors[idx * 3 + 1];
            uint8_t bg_b = bg_colors[idx * 3 + 2];

            out << rgb_to_ansi(fg_r, fg_g, fg_b, mode);
            out << rgb_to_ansi_bg(bg_r, bg_g, bg_b, mode);

            int codepoint = 0x2800 + pattern;
            out << (char)(0xE0 | (codepoint >> 12))
                << (char)(0x80 | ((codepoint >> 6) & 0x3F))
                << (char)(0x80 | (codepoint & 0x3F));
        }

        out << "\033[0m";
        if (cy < cells_y - 1)
            out << '\n';
    }

    return out.str();
}

// Forward declaration for render_hybrid (defined after CUDA wrapper)
static std::string render_hybrid(
    const uint8_t *frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int term_width,
    int term_height,
    ColorMode mode,
    bool keep_aspect_ratio,
    double scale_factor,
    uint8_t detail_level,
    uint8_t threshold_steps,
    double rotation_angle);

// CUDA-accelerated hybrid renderer
// Now supports rotation natively in CUDA
static std::string render_hybrid_cuda_wrapper(
    const uint8_t *frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int term_width,
    int term_height,
    ColorMode mode,
    bool keep_aspect_ratio,
    double scale_factor,
    uint8_t detail_level,
    uint8_t threshold_steps,
    double rotation_angle)
{
    if (!frame_data || frame_width == 0 || frame_height == 0)
        return "";

    std::ostringstream out;
    out.precision(0);
    out << std::fixed;

    // Get rotated dimensions for aspect ratio calculation
    uint32_t rot_width, rot_height;
    get_rotated_dimensions(frame_width, frame_height, rotation_angle, rot_width, rot_height);

    // Calculate scaled dimensions based on rotated image size
    int w_scaled, h_scaled;
    if (keep_aspect_ratio)
    {
        double src_aspect = (double)rot_width / rot_height;
        double term_aspect = (double)(term_width * 2) / (term_height * 4);

        if (src_aspect > term_aspect)
        {
            w_scaled = term_width * 2;
            h_scaled = (int)(w_scaled / src_aspect);
        }
        else
        {
            h_scaled = term_height * 4;
            w_scaled = (int)(h_scaled * src_aspect);
        }
    }
    else
    {
        w_scaled = (int)(rot_width * scale_factor);
        h_scaled = (int)(rot_height * scale_factor);
    }

    w_scaled = std::clamp(w_scaled, 2, term_width * 2);
    h_scaled = std::clamp(h_scaled, 4, term_height * 4);

    int cells_x = w_scaled / 2;
    int cells_y = h_scaled / 4;

    // Allocate output buffers
    std::vector<uint8_t> modes(cells_x * cells_y);
    std::vector<uint8_t> patterns(cells_x * cells_y);
    std::vector<uint8_t> fg_colors(cells_x * cells_y * 3);
    std::vector<uint8_t> bg_colors(cells_x * cells_y * 3);

    // Call CUDA kernel with rotation support
    render_hybrid_cuda(
        frame_data, frame_width, frame_height, frame_stride,
        w_scaled, h_scaled, cells_x, cells_y,
        rotation_angle,
        detail_level, threshold_steps,
        modes.data(), patterns.data(), fg_colors.data(), bg_colors.data());

    // Convert results to ANSI string
    for (int cy = 0; cy < cells_y; cy++)
    {
        for (int cx = 0; cx < cells_x; cx++)
        {
            int idx = cy * cells_x + cx;
            uint8_t cu_mode = modes[idx];
            uint8_t pattern = patterns[idx];

            uint8_t fg_r = fg_colors[idx * 3 + 0];
            uint8_t fg_g = fg_colors[idx * 3 + 1];
            uint8_t fg_b = fg_colors[idx * 3 + 2];

            uint8_t bg_r = bg_colors[idx * 3 + 0];
            uint8_t bg_g = bg_colors[idx * 3 + 1];
            uint8_t bg_b = bg_colors[idx * 3 + 2];

            out << rgb_to_ansi(fg_r, fg_g, fg_b, mode);
            out << rgb_to_ansi_bg(bg_r, bg_g, bg_b, mode);

            if (cu_mode == 1)
            {
                // Braille mode
                int codepoint = 0x2800 + pattern;
                out << (char)(0xE0 | (codepoint >> 12))
                    << (char)(0x80 | ((codepoint >> 6) & 0x3F))
                    << (char)(0x80 | (codepoint & 0x3F));
            }
            else
            {
                // Half block mode
                out << "▀"; // Upper half block
            }
        }

        out << "\033[0m";
        if (cy < cells_y - 1)
            out << '\n';
    }

    return out.str();
}

static std::string render_hybrid(
    const uint8_t *frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int term_width,
    int term_height,
    ColorMode mode,
    bool keep_aspect_ratio,
    double scale_factor,
    uint8_t detail_level,
    uint8_t threshold_steps,
    double rotation_angle)
{

    if (!frame_data || frame_width == 0 || frame_height == 0)
        return "";

    std::ostringstream out;
    out.precision(0);
    out << std::fixed;

    // Get rotated dimensions
    uint32_t rot_width, rot_height;
    get_rotated_dimensions(frame_width, frame_height, rotation_angle, rot_width, rot_height);

    // Use braille dimensions with rotated aspect
    int w_scaled, h_scaled;
    if (keep_aspect_ratio)
    {
        double src_aspect = (double)rot_width / rot_height;
        double term_aspect = (double)(term_width * 2) / (term_height * 4);

        if (src_aspect > term_aspect)
        {
            w_scaled = term_width * 2;
            h_scaled = (int)(w_scaled / src_aspect);
        }
        else
        {
            h_scaled = term_height * 4;
            w_scaled = (int)(h_scaled * src_aspect);
        }
    }
    else
    {
        w_scaled = (int)(rot_width * scale_factor);
        h_scaled = (int)(rot_height * scale_factor);
    }

    w_scaled = std::clamp(w_scaled, 2, term_width * 2);
    h_scaled = std::clamp(h_scaled, 4, term_height * 4);

    int cells_x = w_scaled / 2;
    int cells_y = h_scaled / 4;

    // Calculate kernel size based on detail
    int kernel_size;
    if (detail_level >= 95) kernel_size = 1;
    else if (detail_level >= 80) kernel_size = 2;
    else if (detail_level >= 60) kernel_size = 3;
    else kernel_size = 2;

    // Braille dot positions (2x4 grid)
    int dot_positions[8][2] = {
        {0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {0, 3}, {1, 3}};

    // Edge detection adjacency pairs
    int adjacency_pairs[][2] = {
        {0, 1}, {1, 2}, {3, 4}, {4, 5}, {6, 7},
        {0, 3}, {1, 4}, {2, 5},
        {0, 4}, {1, 3}, {1, 5}, {2, 4}
    };
    double edge_threshold = (detail_level >= 70) ? 30.0 : 50.0;

    for (int cy = 0; cy < cells_y; cy++)
    {
        for (int cx = 0; cx < cells_x; cx++)
        {
            // Sample all 8 dots with rotation
            double lumas[8];
            uint8_t colors[8][3];

            for (int dot = 0; dot < 8; dot++)
            {
                int dot_x = dot_positions[dot][0];
                int dot_y = dot_positions[dot][1];

                int rot_x = (cx * 2 + dot_x) * rot_width / w_scaled;
                int rot_y = (cy * 4 + dot_y) * rot_height / h_scaled;

                // Sample with kernel
                uint32_t r_sum = 0, g_sum = 0, b_sum = 0;
                double weight_sum = 0;

                for (int ky = 0; ky < kernel_size; ky++)
                {
                    for (int kx = 0; kx < kernel_size; kx++)
                    {
                        int px = std::clamp(rot_x + kx - kernel_size / 2, 0, (int)rot_width - 1);
                        int py = std::clamp(rot_y + ky - kernel_size / 2, 0, (int)rot_height - 1);

                        uint8_t r, g, b;
                        sample_rotated_pixel(frame_data, frame_width, frame_height, frame_stride,
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

                colors[dot][0] = r_sum / weight_sum;
                colors[dot][1] = g_sum / weight_sum;
                colors[dot][2] = b_sum / weight_sum;
                lumas[dot] = 0.299 * colors[dot][0] + 0.587 * colors[dot][1] + 0.114 * colors[dot][2];
            }

            // Check for edges
            bool has_edge = false;
            for (auto &pair : adjacency_pairs)
            {
                int i = pair[0];
                int j = pair[1];
                double dr = colors[i][0] - colors[j][0];
                double dg = colors[i][1] - colors[j][1];
                double db = colors[i][2] - colors[j][2];
                double color_dist = sqrt(2.0 * dr * dr + 4.0 * dg * dg + 3.0 * db * db);
                double luma_diff = fabs(lumas[i] - lumas[j]);
                double edge = fmax(color_dist, luma_diff * 2.0);
                if (edge > edge_threshold) {
                    has_edge = true;
                    break;
                }
            }

            if (has_edge)
            {
                // Braille rendering
                double mean_luma = 0;
                for (int i = 0; i < 8; i++) mean_luma += lumas[i];
                mean_luma /= 8.0;

                uint8_t pattern = 0;
                for (int dot = 0; dot < 8; dot++)
                {
                    if (lumas[dot] > mean_luma)
                        pattern |= (1 << dot);
                }

                int lit_count = __builtin_popcount(pattern);
                if (lit_count > 4) pattern = ~pattern;

                uint32_t fg_r = 0, fg_g = 0, fg_b = 0, fg_count = 0;
                uint32_t bg_r = 0, bg_g = 0, bg_b = 0, bg_count = 0;

                for (int i = 0; i < 8; i++)
                {
                    if (pattern & (1 << i))
                    {
                        fg_r += colors[i][0];
                        fg_g += colors[i][1];
                        fg_b += colors[i][2];
                        fg_count++;
                    }
                    else
                    {
                        bg_r += colors[i][0];
                        bg_g += colors[i][1];
                        bg_b += colors[i][2];
                        bg_count++;
                    }
                }

                uint8_t fr = fg_count ? fg_r / fg_count : 128;
                uint8_t fg = fg_count ? fg_g / fg_count : 128;
                uint8_t fb = fg_count ? fg_b / fg_count : 128;
                uint8_t br = bg_count ? bg_r / bg_count : 64;
                uint8_t bgc = bg_count ? bg_g / bg_count : 64;
                uint8_t bb = bg_count ? bg_b / bg_count : 64;

                out << rgb_to_ansi(fr, fg, fb, mode);
                out << rgb_to_ansi_bg(br, bgc, bb, mode);

                int codepoint = 0x2800 + pattern;
                out << (char)(0xE0 | (codepoint >> 12))
                    << (char)(0x80 | ((codepoint >> 6) & 0x3F))
                    << (char)(0x80 | (codepoint & 0x3F));
            }
            else
            {
                // Half-block rendering for flat areas
                uint32_t top_r = 0, top_g = 0, top_b = 0;
                uint32_t bot_r = 0, bot_g = 0, bot_b = 0;

                for (int i : {0, 1, 3, 4})
                {
                    top_r += colors[i][0];
                    top_g += colors[i][1];
                    top_b += colors[i][2];
                }

                for (int i : {2, 5, 6, 7})
                {
                    bot_r += colors[i][0];
                    bot_g += colors[i][1];
                    bot_b += colors[i][2];
                }

                out << rgb_to_ansi(top_r / 4, top_g / 4, top_b / 4, mode);
                out << rgb_to_ansi_bg(bot_r / 4, bot_g / 4, bot_b / 4, mode);
                out << "▀";
            }
        }

        out << "\033[0m";
        if (cy < cells_y - 1)
            out << '\n';
    }

    return out.str();
}

// BLOCKS renderer (2x1 half-blocks, fast and clean)
static std::string render_blocks(
    const uint8_t *frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int term_width,
    int term_height,
    ColorMode mode,
    bool keep_aspect_ratio,
    double scale_factor,
    uint8_t detail_level,
    uint8_t threshold_steps,
    double rotation_angle)
{

    if (!frame_data || frame_width == 0 || frame_height == 0)
        return "";

    std::ostringstream out;
    out.precision(0);
    out << std::fixed;

    // Get rotated dimensions
    uint32_t rot_width, rot_height;
    get_rotated_dimensions(frame_width, frame_height, rotation_angle, rot_width, rot_height);

    // Calculate scaled dimensions using rotated dimensions
    int w_scaled, h_scaled;
    if (keep_aspect_ratio)
    {
        double src_aspect = (double)rot_width / rot_height;
        double term_aspect = (double)term_width / (term_height * 2);

        if (src_aspect > term_aspect)
        {
            w_scaled = term_width;
            h_scaled = (int)(w_scaled / src_aspect);
        }
        else
        {
            h_scaled = term_height * 2;
            w_scaled = (int)(h_scaled * src_aspect);
        }
    }
    else
    {
        w_scaled = (int)(rot_width * scale_factor);
        h_scaled = (int)(rot_height * scale_factor);
    }

    w_scaled = std::clamp(w_scaled, 1, term_width);
    h_scaled = std::clamp(h_scaled, 2, term_height * 2);

    int blocks_x = w_scaled;
    int blocks_y = h_scaled / 2;

    // Adaptive sampling kernel
    int kernel_size;
    if (detail_level >= 95)
    {
        kernel_size = 1; // Pixel perfect
    }
    else if (detail_level >= 70)
    {
        kernel_size = 2;
    }
    else if (detail_level >= 40)
    {
        kernel_size = 3;
    }
    else
    {
        kernel_size = 4;
    }

    // Render blocks
    for (int by = 0; by < blocks_y; by++)
    {
        for (int bx = 0; bx < blocks_x; bx++)
        {
            // Calculate source coordinates in rotated space
            int rot_x = (bx * rot_width) / w_scaled;
            int rot_y_top = ((by * 2) * rot_height) / h_scaled;
            int rot_y_bot = ((by * 2 + 1) * rot_height) / h_scaled;

            // Sample with kernel
            uint32_t r_top = 0, g_top = 0, b_top = 0;
            uint32_t r_bot = 0, g_bot = 0, b_bot = 0;
            double weight_sum = 0;

            for (int ky = 0; ky < kernel_size; ky++)
            {
                for (int kx = 0; kx < kernel_size; kx++)
                {
                    // Gaussian-like weight
                    double dist = sqrt(kx * kx + ky * ky);
                    double sigma = kernel_size * 0.5;
                    double weight = exp(-dist * dist / (2.0 * sigma * sigma));

                    // Top half - sample with rotation
                    int px_top = std::clamp(rot_x + kx, 0, (int)rot_width - 1);
                    int py_top = std::clamp(rot_y_top + ky, 0, (int)rot_height - 1);
                    uint8_t r, g, b;
                    sample_rotated_pixel(frame_data, frame_width, frame_height, frame_stride,
                                        px_top, py_top, rot_width, rot_height, rotation_angle, r, g, b);
                    r_top += r * weight;
                    g_top += g * weight;
                    b_top += b * weight;

                    // Bottom half - sample with rotation
                    int px_bot = std::clamp(rot_x + kx, 0, (int)rot_width - 1);
                    int py_bot = std::clamp(rot_y_bot + ky, 0, (int)rot_height - 1);
                    sample_rotated_pixel(frame_data, frame_width, frame_height, frame_stride,
                                        px_bot, py_bot, rot_width, rot_height, rotation_angle, r, g, b);
                    r_bot += r * weight;
                    g_bot += g * weight;
                    b_bot += b * weight;

                    weight_sum += weight;
                }
            }

            r_top /= weight_sum;
            g_top /= weight_sum;
            b_top /= weight_sum;
            r_bot /= weight_sum;
            g_bot /= weight_sum;
            b_bot /= weight_sum;

            out << rgb_to_ansi(r_top, g_top, b_top, mode);
            out << rgb_to_ansi_bg(r_bot, g_bot, b_bot, mode);
            out << "▀";
        }

        out << "\033[0m";
        if (by < blocks_y - 1)
            out << '\n';
    }

    return out.str();
}

// ASCII renderer (character brightness mapping)
static std::string render_ascii(
    const uint8_t *frame_data,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_stride,
    int term_width,
    int term_height,
    ColorMode mode,
    bool keep_aspect_ratio,
    double scale_factor,
    uint8_t detail_level,
    uint8_t threshold_steps,
    double rotation_angle)
{

    if (!frame_data || frame_width == 0 || frame_height == 0)
        return "";

    std::ostringstream out;
    out.precision(0);
    out << std::fixed;

    // ASCII brightness ramp
    const char *chars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
    int char_count = strlen(chars);

    // Get rotated dimensions
    uint32_t rot_width, rot_height;
    get_rotated_dimensions(frame_width, frame_height, rotation_angle, rot_width, rot_height);

    // Calculate scaled dimensions using rotated dimensions
    int w_scaled, h_scaled;
    if (keep_aspect_ratio)
    {
        double src_aspect = (double)rot_width / rot_height;
        double term_aspect = (double)term_width / term_height;

        if (src_aspect > term_aspect)
        {
            w_scaled = term_width;
            h_scaled = (int)(w_scaled / src_aspect);
        }
        else
        {
            h_scaled = term_height;
            w_scaled = (int)(h_scaled * src_aspect);
        }
    }
    else
    {
        w_scaled = (int)(rot_width * scale_factor);
        h_scaled = (int)(rot_height * scale_factor);
    }

    w_scaled = std::clamp(w_scaled, 1, term_width);
    h_scaled = std::clamp(h_scaled, 1, term_height);

    // Adaptive sampling
    int kernel_size = (detail_level >= 95) ? 1 : (2 + (100 - detail_level) / 25);

    // Render ASCII art
    for (int ay = 0; ay < h_scaled; ay++)
    {
        for (int ax = 0; ax < w_scaled; ax++)
        {
            int rot_x = (ax * rot_width) / w_scaled;
            int rot_y = (ay * rot_height) / h_scaled;

            // Sample with kernel
            uint32_t r_sum = 0, g_sum = 0, b_sum = 0;
            double weight_sum = 0;

            for (int ky = 0; ky < kernel_size; ky++)
            {
                for (int kx = 0; kx < kernel_size; kx++)
                {
                    int px = std::clamp(rot_x + kx, 0, (int)rot_width - 1);
                    int py = std::clamp(rot_y + ky, 0, (int)rot_height - 1);

                    uint8_t r, g, b;
                    sample_rotated_pixel(frame_data, frame_width, frame_height, frame_stride,
                                        px, py, rot_width, rot_height, rotation_angle, r, g, b);

                    double dist = sqrt(kx * kx + ky * ky);
                    double sigma = kernel_size * 0.5;
                    double weight = exp(-dist * dist / (2.0 * sigma * sigma));

                    r_sum += r * weight;
                    g_sum += g * weight;
                    b_sum += b * weight;
                    weight_sum += weight;
                }
            }

            uint8_t r = r_sum / weight_sum;
            uint8_t g = g_sum / weight_sum;
            uint8_t b = b_sum / weight_sum;

            // Calculate luminance
            double luma = 0.299 * r + 0.587 * g + 0.114 * b;
            int char_idx = std::clamp((int)((luma / 255.0) * (char_count - 1)), 0, char_count - 1);

            out << rgb_to_ansi(r, g, b, mode) << chars[char_idx];
        }

        out << "\033[0m";
        if (ay < h_scaled - 1)
            out << '\n';
    }

    return out.str();
}

static void capture_thread(int output_index, int fps) {
    if (output_index >= (int)outputs.size())
        return;

    const long frame_delay = 1000000 / fps;
    
    std::cerr << "[CAPTURE] Thread started for output " << output_index
              << " at " << fps << " FPS using " 
              << (capture_backend == PIPEWIRE ? "PipeWire" : "wlr-screencopy") << "\n";

    if (capture_backend == PIPEWIRE) {
        // PipeWire capture loop
        auto &pw_cap = pipewire_captures[output_index];
        
        if (!pw_cap) {
            std::cerr << "[CAPTURE] Output " << output_index << " not initialized\n";
            return;
        }
        
        auto last_frame_time = std::chrono::steady_clock::now();
        
        while (running) {
            auto frame_start = std::chrono::steady_clock::now();
            
            std::vector<uint8_t> frame_data;
            uint32_t width, height, stride;
            
            if (pw_cap->get_frame(frame_data, width, height, stride)) {
                // Copy to output buffer
                std::lock_guard<std::mutex> lock(*output_mutexes[output_index]);
                Capture &cap = output_captures[output_index];
                
                cap.back_buffer = std::move(frame_data);
                cap.width = width;
                cap.height = height;
                cap.stride = stride;
                cap.format = 0; // BGRA
                cap.back_ready = true;
                cap.timestamp = std::chrono::steady_clock::now();
                
                output_ready[output_index] = true;
                frame_ready_cvs[output_index]->notify_all();
                
                static int frame_log_count = 0;
            } else {
                static int no_frame_count = 0;
                if (++no_frame_count % 300 == 1) {
                    std::cerr << "[CAPTURE] No frame available for output " << output_index 
                              << " (" << no_frame_count << " attempts)\n";
                }
            }
            
            // Frame pacing
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                now - last_frame_time).count();
            
            if (elapsed < frame_delay) {
                usleep(frame_delay - elapsed);
            }
            
            last_frame_time = std::chrono::steady_clock::now();
        }
    } else {
        // Original wlr-screencopy
        wl_output *output = outputs[output_index];
        
        CaptureContext ctx;
        ctx.output_index = output_index;
        ctx.capture = &output_captures[output_index];
        ctx.mutex = output_mutexes[output_index].get();
        ctx.ready = &output_ready[output_index];

        auto last_frame_time = std::chrono::steady_clock::now();
        int consecutive_timeouts = 0;

        {
            std::lock_guard<std::mutex> lock(*ctx.mutex);
            auto *frame = zwlr_screencopy_manager_v1_capture_output(manager, 1, output);
            zwlr_screencopy_frame_v1_add_listener(frame, &frame_listener, &ctx);
            wl_display_flush(display);
            ctx.capture->frame_in_flight = true;
        }

        while (running) {
            auto frame_start = std::chrono::steady_clock::now();

            {
                std::unique_lock<std::mutex> lock(*output_mutexes[output_index]);

                bool got_frame = frame_ready_cvs[output_index]->wait_for(
                    lock,
                    std::chrono::milliseconds(100),
                    [&] { return output_ready[output_index] || !running; });

                if (!running) break;

                if (!got_frame) {
                    consecutive_timeouts++;
                    if (consecutive_timeouts > 5) {
                        std::cerr << "[CAPTURE] Multiple timeouts on output " << output_index << "\n";
                        consecutive_timeouts = 0;
                    }
                    continue;
                }

                consecutive_timeouts = 0;
                output_ready[output_index] = false;

                if (ctx.capture->front_ready && ctx.capture->front_data) {
                    if (!ctx.capture->back_ready ||
                        ctx.capture->back_buffer.size() != ctx.capture->size) {
                        ctx.capture->back_buffer.resize(ctx.capture->size);
                    }

                    memcpy(ctx.capture->back_buffer.data(),
                           ctx.capture->front_data,
                           ctx.capture->size);

                    ctx.capture->back_ready = true;
                    ctx.capture->front_ready = false;
                }
            }

            {
                std::lock_guard<std::mutex> lock(*ctx.mutex);

                if (!ctx.capture->frame_in_flight) {
                    auto *frame = zwlr_screencopy_manager_v1_capture_output(manager, 1, output);
                    zwlr_screencopy_frame_v1_add_listener(frame, &frame_listener, &ctx);
                    wl_display_flush(display);
                    ctx.capture->frame_in_flight = true;
                }
            }

            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                now - last_frame_time).count();

            if (elapsed < frame_delay) {
                long sleep_time = frame_delay - elapsed;

                if (sleep_time > 2000) {
                    usleep(sleep_time - 1000);

                    while (true) {
                        now = std::chrono::steady_clock::now();
                        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                            now - last_frame_time).count();
                        if (elapsed >= frame_delay) break;
                    }
                } else {
                    while (true) {
                        now = std::chrono::steady_clock::now();
                        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                            now - last_frame_time).count();
                        if (elapsed >= frame_delay) break;
                    }
                }
            }

            last_frame_time = std::chrono::steady_clock::now();
        }
    }

    std::cerr << "[CAPTURE] Thread stopped for output " << output_index << "\n";
}

static void send_thread(int socket, std::shared_ptr<SendQueue> queue)
{
    std::cerr << "[SEND] Thread started\n";

    while (queue->running)
    {
        std::unique_lock<std::mutex> lock(queue->mutex);

        // Wait for a frame to be ready
        queue->cv.wait_for(lock, std::chrono::milliseconds(10), [&]
                           { return queue->frame_ready || !queue->running; });

        if (!queue->running)
            break;
        if (!queue->frame_ready)
            continue;

        // Take the frame and mark as not ready
        auto message = std::move(queue->latest_frame);
        queue->frame_ready = false;
        lock.unlock();

        // Send without holding lock
        size_t total_sent = 0;
        while (total_sent < message.size() && queue->running)
        {
            ssize_t sent = send(socket, message.data() + total_sent,
                                message.size() - total_sent, MSG_NOSIGNAL);

            if (sent > 0)
            {
                total_sent += sent;
            }
            else if (sent < 0)
            {
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                {
                    usleep(100);
                    continue;
                }
                else if (errno == EPIPE || errno == ECONNRESET)
                {
                    std::cerr << "[SEND] Connection lost\n";
                    queue->running = false;
                    return;
                }
                else
                {
                    std::cerr << "[SEND] Error: " << strerror(errno) << "\n";
                    queue->running = false;
                    return;
                }
            }
        }
    }

    std::cerr << "[SEND] Thread stopped\n";
}

static std::string generate_client_id(const sockaddr_in &addr)
{
    // This is now just a fallback - actual ID comes from SessionID message
    return std::string(inet_ntoa(addr.sin_addr)) + ":fallback";
}

// ADD helper to receive session ID:
static bool receive_session_id(int socket, std::string &session_id)
{
    MessageType type;
    ssize_t n = recv(socket, &type, sizeof(type), 0);

    if (n != sizeof(type))
    {
        std::cerr << "Failed to receive session ID type\n";
        return false;
    }

    if (type != MessageType::SESSION_ID)
    {
        std::cerr << "Expected SESSION_ID message, got " << (int)type << "\n";
        return false;
    }

    SessionID sid;
    n = recv(socket, &sid, sizeof(sid), 0);

    if (n != sizeof(sid))
    {
        std::cerr << "Failed to receive session ID data\n";
        return false;
    }

    sid.uuid[36] = '\0'; // Ensure null termination
    session_id = sid.uuid;

    std::cerr << "Received session ID: " << session_id << "\n";
    return true;
}

static std::vector<uint8_t> compress_frame(const std::vector<uint8_t> &data,
                                           int compression_level)
{
    int max_size = LZ4_compressBound(data.size());
    std::vector<uint8_t> compressed(max_size);
    int comp_size;

    if (compression_level == 0)
    {
        // Fast LZ4 (default)
        comp_size = LZ4_compress_default(
            (const char *)data.data(),
            (char *)compressed.data(),
            data.size(),
            max_size);
    }
    else
    {
        // LZ4-HC (high compression)
        comp_size = LZ4_compress_HC(
            (const char *)data.data(),
            (char *)compressed.data(),
            data.size(),
            max_size,
            compression_level // 1-12, higher = better compression, slower
        );
    }

    if (comp_size <= 0)
    {
        std::cerr << "[COMPRESS] Failed, sending uncompressed\n";
        return data;
    }

    compressed.resize(comp_size);

    // Stats
    total_bytes_original += data.size();
    total_bytes_compressed += comp_size;
    
    if (total_frames_sent % 100 == 0)
    {
        double saved = 100.0 * (1.0 - (double)total_bytes_compressed / total_bytes_original);
        std::cerr << "[COMPRESS] Level " << compression_level
                  << " | Saved: " << std::fixed << std::setprecision(2) << saved << "% over "
                  << total_frames_sent << " frames\n";
    }

    return compressed;
}

static std::vector<uint8_t> encode_delta_frame(
    const std::string &old_frame,
    const std::string &new_frame,
    bool compress,
    int compression_level)
{

    std::vector<uint8_t> delta;

    if (old_frame.empty() || old_frame.size() != new_frame.size())
    {
        // Size mismatch or no previous frame - can't delta encode
        return delta;
    }

    // Find all changed regions
    std::vector<FrameChange> changes;
    size_t i = 0;
    size_t frame_size = new_frame.size();

    while (i < frame_size)
    {
        // Skip identical bytes
        while (i < frame_size && old_frame[i] == new_frame[i])
        {
            i++;
        }

        if (i >= frame_size)
            break;

        // Found a change - find the end
        size_t start = i;
        while (i < frame_size && old_frame[i] != new_frame[i])
        {
            i++;
        }

        size_t length = i - start;

        // Split large changes into chunks to avoid overflow
        while (length > 0)
        {
            uint16_t chunk_len = std::min(length, (size_t)65535);
            changes.push_back({(uint32_t)start, chunk_len});
            start += chunk_len;
            length -= chunk_len;
        }
    }

    if (changes.empty())
    {
        // No changes - return empty delta to signal identical frame
        return delta;
    }

    // Calculate uncompressed delta size
    size_t delta_size = sizeof(DeltaFrameHeader);
    for (const auto &change : changes)
    {
        delta_size += sizeof(FrameChange) + change.length;
    }

    // Only use delta if it's significantly smaller (less than 75% of full frame)
    if (delta_size > new_frame.size() * 0.75)
    {
        return {}; // Not worth it
    }

    // Build delta message
    delta.reserve(delta_size);

    DeltaFrameHeader header;
    header.num_changes = changes.size();
    header.base_frame_size = old_frame.size();

    const uint8_t *header_bytes = reinterpret_cast<const uint8_t *>(&header);
    delta.insert(delta.end(), header_bytes, header_bytes + sizeof(header));

    for (const auto &change : changes)
    {
        // Add change header
        const uint8_t *change_bytes = reinterpret_cast<const uint8_t *>(&change);
        delta.insert(delta.end(), change_bytes, change_bytes + sizeof(change));

        // Add changed data
        const uint8_t *data = reinterpret_cast<const uint8_t *>(
            new_frame.data() + change.offset);
        delta.insert(delta.end(), data, data + change.length);
    }

    // Optionally compress the delta
    if (compress && delta.size() > 1024)
    {
        std::vector<uint8_t> compressed = compress_frame(delta, compression_level);

        // Only use compression if it actually helps
        if (compressed.size() < delta.size() * 0.9)
        {
            return compressed;
        }
    }

    return delta;
}

static void on_process_microphone_source(void *userdata)
{
    MicrophoneVirtualSource *src = static_cast<MicrophoneVirtualSource *>(userdata);

    pw_buffer *b = pw_stream_dequeue_buffer(src->stream);
    if (!b)
        return;

    spa_buffer *buf = b->buffer;
    if (!buf->datas[0].data)
    {
        pw_stream_queue_buffer(src->stream, b);
        return;
    }

    uint8_t *dst = static_cast<uint8_t *>(buf->datas[0].data);
    uint32_t max_size = buf->datas[0].maxsize;
    uint32_t size = 0;

    std::lock_guard<std::mutex> lock(src->mutex);

    if (!src->microphone_queue.empty())
    {
        auto &microphone_data = src->microphone_queue.front();
        size = std::min((uint32_t)microphone_data.size(), max_size);
        memcpy(dst, microphone_data.data(), size);
        src->microphone_queue.pop();
    }
    else
    {
        // Silence if no data
        size = max_size;
        memset(dst, 0, size);
    }

    buf->datas[0].chunk->offset = 0;
    buf->datas[0].chunk->stride = src->format.channels * sizeof(float);
    buf->datas[0].chunk->size = size;

    pw_stream_queue_buffer(src->stream, b);
}

static const pw_stream_events microphone_source_events = {
    .version = PW_VERSION_STREAM_EVENTS,
    .process = on_process_microphone_source,
};

static bool init_microphone_virtual_source(const AudioFormat &fmt)
{
    if (!feature_microphone)
        return true;

    microphone_virtual_source.format = fmt;
    microphone_virtual_source.loop = pw_thread_loop_new("mic-virtual-source", nullptr);
    if (!microphone_virtual_source.loop)
    {
        std::cerr << "[MICROPHONE IN] Failed to create thread loop\n";
        return false;
    }

    pw_thread_loop_lock(microphone_virtual_source.loop);

    pw_context *context = pw_context_new(
        pw_thread_loop_get_loop(microphone_virtual_source.loop), nullptr, 0);

    if (!context)
    {
        std::cerr << "[MICROPHONE IN] Failed to create context\n";
        pw_thread_loop_unlock(microphone_virtual_source.loop);
        return false;
    }

    // KEY CHANGE: This creates a virtual SOURCE (input device), not a playback sink
    microphone_virtual_source.stream = pw_stream_new_simple(
        pw_thread_loop_get_loop(microphone_virtual_source.loop),
        "wayterm-mirror-virtual-microphone",
        pw_properties_new(
            PW_KEY_MEDIA_TYPE, "Audio",
            PW_KEY_MEDIA_CATEGORY, "Source", // Changed from "Playback" to "Source"
            PW_KEY_MEDIA_ROLE, "Communication",
            PW_KEY_NODE_NAME, "waytermirror_virtual_mic",
            PW_KEY_NODE_DESCRIPTION, "Wayterm Virtual Microphone",
            nullptr),
        &microphone_source_events,
        &microphone_virtual_source);

    if (!microphone_virtual_source.stream)
    {
        std::cerr << "[MICROPHONE IN] Failed to create stream\n";
        pw_thread_loop_unlock(microphone_virtual_source.loop);
        return false;
    }

    uint8_t buffer[1024];
    spa_pod_builder b = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));

    const spa_pod *params[1];
    auto spa_audio_info_raw = SPA_AUDIO_INFO_RAW_INIT(
            .format = (fmt.format == 0) ? SPA_AUDIO_FORMAT_S16 : SPA_AUDIO_FORMAT_F32,
            .rate = fmt.sample_rate,
            .channels = fmt.channels);

    params[0] = spa_format_audio_raw_build(&b, SPA_PARAM_EnumFormat,
                                           &spa_audio_info_raw);

    // KEY CHANGE: Direction is OUTPUT because we're outputting TO the system as a source
    pw_stream_connect(microphone_virtual_source.stream,
                      PW_DIRECTION_OUTPUT, // We output audio data to become an input source
                      PW_ID_ANY,
                      static_cast<pw_stream_flags>(
                          PW_STREAM_FLAG_MAP_BUFFERS |
                          PW_STREAM_FLAG_RT_PROCESS),
                      params, 1);

    pw_thread_loop_unlock(microphone_virtual_source.loop);
    pw_thread_loop_start(microphone_virtual_source.loop);

    std::cerr << "[MICROPHONE IN] Virtual microphone source created (" << fmt.sample_rate
              << "Hz " << (int)fmt.channels << "ch)\n";
    std::cerr << "[MICROPHONE IN] Applications can now select 'Wayterm Virtual Microphone' as input\n";
    return true;
}

static void cleanup_microphone_virtual_source()
{
    if (!feature_microphone)
        return;

    microphone_virtual_source.running = false;

    if (microphone_virtual_source.stream)
    {
        pw_stream_destroy(microphone_virtual_source.stream);
    }

    if (microphone_virtual_source.loop)
    {
        pw_thread_loop_stop(microphone_virtual_source.loop);
        pw_thread_loop_destroy(microphone_virtual_source.loop);
    }

    std::cerr << "[MICROPHONE IN] Virtual source cleaned up\n";
}

static void on_process_audio(void *userdata)
{
    AudioCapture *cap = static_cast<AudioCapture *>(userdata);

    pw_buffer *b = pw_stream_dequeue_buffer(cap->stream);
    if (!b)
        return;

    spa_buffer *buf = b->buffer;
    if (!buf->datas[0].data)
    {
        pw_stream_queue_buffer(cap->stream, b);
        return;
    }

    uint8_t *src = static_cast<uint8_t *>(buf->datas[0].data);
    uint32_t size = buf->datas[0].chunk->size;

    if (size > 0)
    {
        std::vector<uint8_t> audio_data(src, src + size);

        std::lock_guard<std::mutex> lock(cap->mutex);
        cap->audio_queue.push(std::move(audio_data));

        while (cap->audio_queue.size() > 10)
        {
            cap->audio_queue.pop();
        }
    }

    pw_stream_queue_buffer(cap->stream, b);
}

static const pw_stream_events stream_events = {
    .version = PW_VERSION_STREAM_EVENTS,
    .process = on_process_audio,
};

static bool init_audio_capture()
{
    if (!feature_audio)
        return true;

    pw_init(nullptr, nullptr);

    audio_capture.loop = pw_thread_loop_new("audio-capture", nullptr);
    if (!audio_capture.loop)
    {
        std::cerr << "[AUDIO] Failed to create thread loop\n";
        return false;
    }

    pw_thread_loop_lock(audio_capture.loop);

    pw_context *context = pw_context_new(
        pw_thread_loop_get_loop(audio_capture.loop), nullptr, 0);

    if (!context)
    {
        std::cerr << "[AUDIO] Failed to create context\n";
        pw_thread_loop_unlock(audio_capture.loop);
        return false;
    }

    // FIX: Capture system audio OUTPUT (what you hear), not microphone
    audio_capture.stream = pw_stream_new_simple(
        pw_thread_loop_get_loop(audio_capture.loop),
        "wayterm-mirror-system-audio",
        pw_properties_new(
            PW_KEY_MEDIA_TYPE, "Audio",
            PW_KEY_MEDIA_CATEGORY, "Capture",
            PW_KEY_MEDIA_ROLE, "Music",
            // ADD: Monitor default sink output
            PW_KEY_NODE_TARGET, "@DEFAULT_AUDIO_SINK@.monitor",
            PW_KEY_STREAM_CAPTURE_SINK, "true",
            nullptr),
        &stream_events,
        &audio_capture);

    if (!audio_capture.stream)
    {
        std::cerr << "[AUDIO] Failed to create stream\n";
        pw_thread_loop_unlock(audio_capture.loop);
        return false;
    }

    uint8_t buffer[1024];
    spa_pod_builder b = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));

    const spa_pod *params[1];
    auto spa_audio_info_raw = SPA_AUDIO_INFO_RAW_INIT(
            .format = SPA_AUDIO_FORMAT_F32,
            .rate = 48000,
            .channels = 2);
    params[0] = spa_format_audio_raw_build(&b, SPA_PARAM_EnumFormat,
                                           &spa_audio_info_raw);

    pw_stream_connect(audio_capture.stream,
                      PW_DIRECTION_INPUT,
                      PW_ID_ANY,
                      static_cast<pw_stream_flags>(
                          PW_STREAM_FLAG_AUTOCONNECT |
                          PW_STREAM_FLAG_MAP_BUFFERS |
                          PW_STREAM_FLAG_RT_PROCESS),
                      params, 1);

    pw_thread_loop_unlock(audio_capture.loop);
    pw_thread_loop_start(audio_capture.loop);

    std::cerr << "[AUDIO OUT] System audio capture initialized (monitoring default sink)\n";
    std::cerr << "[AUDIO OUT] Sending: 48kHz stereo F32LE from system audio output\n";
    return true;
}

static void cleanup_audio_capture()
{
    if (!feature_audio)
        return;

    audio_capture.running = false;

    if (audio_capture.stream)
    {
        pw_stream_destroy(audio_capture.stream);
    }

    if (audio_capture.loop)
    {
        pw_thread_loop_stop(audio_capture.loop);
        pw_thread_loop_destroy(audio_capture.loop);
    }

    pw_deinit();
    std::cerr << "[AUDIO] Capture cleaned up\n";
}

static void audio_thread(int client_socket, std::string session_id)
{
    if (!feature_audio)
    {
        close(client_socket);
        return;
    }

    std::cerr << "[AUDIO] Client connected: " << session_id << "\n";

    // Send format
    {
        MessageType type = MessageType::AUDIO_FORMAT;
        send(client_socket, &type, sizeof(type), MSG_NOSIGNAL);
        send(client_socket, &audio_capture.format, sizeof(audio_capture.format), MSG_NOSIGNAL);
    }

    while (running && audio_capture.running)
    {
        std::vector<uint8_t> audio_data;

        {
            std::lock_guard<std::mutex> lock(audio_capture.mutex);
            if (!audio_capture.audio_queue.empty())
            {
                audio_data = std::move(audio_capture.audio_queue.front());
                audio_capture.audio_queue.pop();
            }
        }

        if (!audio_data.empty())
        {
            MessageType type = MessageType::AUDIO_DATA;
            AudioDataHeader header;
            header.size = audio_data.size();
            header.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now().time_since_epoch())
                                      .count();

            if (send(client_socket, &type, sizeof(type), MSG_NOSIGNAL) != sizeof(type) ||
                send(client_socket, &header, sizeof(header), MSG_NOSIGNAL) != sizeof(header) ||
                send(client_socket, audio_data.data(), audio_data.size(), MSG_NOSIGNAL) != (ssize_t)audio_data.size())
            {
                std::cerr << "[AUDIO] Send failed\n";
                break;
            }
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    close(client_socket);
    std::cerr << "[AUDIO] Client disconnected: " << session_id << "\n";
}

static void accept_audio_thread(int server_socket)
{
    if (!feature_audio)
        return;

    std::cerr << "[AUDIO] Accept thread started\n";

    while (running)
    {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);

        int client_socket = accept(server_socket, (sockaddr *)&client_addr, &client_len);
        if (client_socket < 0)
        {
            if (running)
                std::cerr << "[AUDIO] Accept failed\n";
            continue;
        }

        std::string session_id;
        MessageType type;
        SessionID sid;

        if (recv(client_socket, &type, sizeof(type), 0) != sizeof(type) ||
            type != MessageType::SESSION_ID ||
            recv(client_socket, &sid, sizeof(sid), 0) != sizeof(sid))
        {
            close(client_socket);
            continue;
        }

        sid.uuid[36] = '\0';
        session_id = sid.uuid;

        std::thread(audio_thread, client_socket, session_id).detach();
    }

    std::cerr << "[AUDIO] Accept thread stopped\n";
}

static void microphone_thread(int client_socket, std::string session_id)
{
    if (!feature_microphone)
    {
        close(client_socket);
        return;
    }

    std::cerr << "[MICROPHONE IN] Client connected: " << session_id << "\n";

    // Receive format
    MessageType type;
    AudioFormat fmt;

    if (recv(client_socket, &type, sizeof(type), 0) != sizeof(type) ||
        type != MessageType::MICROPHONE_FORMAT ||
        recv(client_socket, &fmt, sizeof(fmt), 0) != sizeof(fmt))
    {
        std::cerr << "[MICROPHONE IN] Failed to receive format\n";
        close(client_socket);
        return;
    }

    if (!init_microphone_virtual_source(fmt))
    { // CHANGED
        std::cerr << "[MICROPHONE IN] Failed to init virtual source\n";
        close(client_socket);
        return;
    }

    while (running && microphone_virtual_source.running)
    { // CHANGED
        if (recv(client_socket, &type, sizeof(type), 0) != sizeof(type))
            break;

        if (type != MessageType::MICROPHONE_DATA)
        {
            std::cerr << "[MICROPHONE IN] Unexpected message type\n";
            continue;
        }

        AudioDataHeader header;
        if (recv(client_socket, &header, sizeof(header), 0) != sizeof(header))
            break;

        std::vector<uint8_t> microphone_data(header.size);
        size_t total = 0;
        while (total < header.size)
        {
            ssize_t n = recv(client_socket, microphone_data.data() + total,
                             header.size - total, 0);
            if (n <= 0)
                goto cleanup;
            total += n;
        }

        {
            std::lock_guard<std::mutex> lock(microphone_virtual_source.mutex);           // CHANGED
            microphone_virtual_source.microphone_queue.push(std::move(microphone_data)); // CHANGED

            // Prevent queue buildup
            while (microphone_virtual_source.microphone_queue.size() > 10)
            { // CHANGED
                microphone_virtual_source.microphone_queue.pop();
            }
        }
    }

cleanup:
    close(client_socket);
    std::cerr << "[MICROPHONE IN] Client disconnected: " << session_id << "\n";
}

static void accept_microphone_thread(int server_socket)
{
    if (!feature_microphone)
        return;

    std::cerr << "[MICROPHONE IN] Accept thread started\n";

    while (running)
    {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);

        int client_socket = accept(server_socket, (sockaddr *)&client_addr, &client_len);
        if (client_socket < 0)
        {
            if (running)
                std::cerr << "[MICROPHONE IN] Accept failed\n";
            continue;
        }

        std::string session_id;
        MessageType type;
        SessionID sid;

        if (recv(client_socket, &type, sizeof(type), 0) != sizeof(type) ||
            type != MessageType::SESSION_ID ||
            recv(client_socket, &sid, sizeof(sid), 0) != sizeof(sid))
        {
            close(client_socket);
            continue;
        }

        sid.uuid[36] = '\0';
        session_id = sid.uuid;

        std::thread(microphone_thread, client_socket, session_id).detach();
    }

    std::cerr << "[MICROPHONE IN] Accept thread stopped\n";
}

static void apply_zoom_transform(
    const uint8_t *src_data,
    uint32_t src_width,
    uint32_t src_height,
    uint32_t src_stride,
    std::vector<uint8_t> &dst_data,
    uint32_t &dst_width,
    uint32_t &dst_height,
    uint32_t &dst_stride,
    const ZoomState &zoom)
{

    std::lock_guard<std::mutex> lock(const_cast<ZoomState &>(zoom).mutex);

    if (!zoom.enabled || zoom.zoom_level <= 1.0)
    {
        // No zoom - copy original
        dst_width = src_width;
        dst_height = src_height;
        dst_stride = src_stride;
        dst_data.resize(src_height * src_stride);
        memcpy(dst_data.data(), src_data, src_height * src_stride);
        return;
    }

    // Calculate zoomed region bounds
    double inv_zoom = 1.0 / zoom.zoom_level;
    int region_width = (int)(zoom.view_width * inv_zoom);
    int region_height = (int)(zoom.view_height * inv_zoom);

    // Center point from mouse position (client sends coordinates in screen space)
    // No scaling needed - client was already told the screen dimensions via SCREEN_INFO
    int center_x = zoom.center_x;
    int center_y = zoom.center_y;

    std::cerr << "[ZOOM] Applying zoom: level=" << zoom.zoom_level
              << " center=(" << center_x << "," << center_y << ")"
              << " follow_mouse=" << zoom.follow_mouse << "\n";

    // Calculate source region
    int src_x = center_x - region_width / 2;
    int src_y = center_y - region_height / 2;

    // Clamp to screen bounds
    src_x = std::clamp(src_x, 0, (int)src_width - region_width);
    src_y = std::clamp(src_y, 0, (int)src_height - region_height);

    // Output dimensions match viewport
    dst_width = zoom.view_width;
    dst_height = zoom.view_height;
    dst_stride = dst_width * 4; // BGRA

    dst_data.resize(dst_height * dst_stride);

    // High-quality bilinear interpolation zoom
    for (uint32_t dy = 0; dy < dst_height; dy++)
    {
        for (uint32_t dx = 0; dx < dst_width; dx++)
        {
            // Map destination pixel to source coordinates
            double sx = src_x + (dx * inv_zoom);
            double sy = src_y + (dy * inv_zoom);

            // Bilinear interpolation
            int sx0 = (int)sx;
            int sy0 = (int)sy;
            int sx1 = std::min(sx0 + 1, (int)src_width - 1);
            int sy1 = std::min(sy0 + 1, (int)src_height - 1);

            double fx = sx - sx0;
            double fy = sy - sy0;

            // Clamp source coordinates
            sx0 = std::clamp(sx0, 0, (int)src_width - 1);
            sy0 = std::clamp(sy0, 0, (int)src_height - 1);

            // Sample 4 pixels
            const uint8_t *p00 = src_data + sy0 * src_stride + sx0 * 4;
            const uint8_t *p10 = src_data + sy0 * src_stride + sx1 * 4;
            const uint8_t *p01 = src_data + sy1 * src_stride + sx0 * 4;
            const uint8_t *p11 = src_data + sy1 * src_stride + sx1 * 4;

            // Interpolate each channel
            uint8_t *dst = dst_data.data() + dy * dst_stride + dx * 4;
            for (int c = 0; c < 4; c++)
            {
                double top = p00[c] * (1.0 - fx) + p10[c] * fx;
                double bot = p01[c] * (1.0 - fx) + p11[c] * fx;
                dst[c] = (uint8_t)(top * (1.0 - fy) + bot * fy);
            }
        }
    }
}

// ZOOM: Update smooth panning in zoom state
static void update_zoom_smooth_pan(ZoomState &zoom)
{
    if (!zoom.smooth_pan || !zoom.enabled)
        return;

    std::lock_guard<std::mutex> lock(zoom.mutex);

    // Smoothly interpolate center position
    int dx = (zoom.target_center_x - zoom.center_x);
    int dy = (zoom.target_center_y - zoom.center_y);

    if (abs(dx) > 0 || abs(dy) > 0)
    {
        int step = zoom.pan_speed;

        if (abs(dx) > step)
        {
            zoom.center_x += (dx > 0) ? step : -step;
        }
        else
        {
            zoom.center_x = zoom.target_center_x;
        }

        if (abs(dy) > step)
        {
            zoom.center_y += (dy > 0) ? step : -step;
        }
        else
        {
            zoom.center_y = zoom.target_center_y;
        }
    }
}

static void handle_frame_client(int client_socket, sockaddr_in client_addr)
{
    std::cerr << "[FRAME] Client connected from " << inet_ntoa(client_addr.sin_addr) << "\n";

    // FIRST: Receive session ID
    std::string session_id;
    if (!receive_session_id(client_socket, session_id))
    {
        std::cerr << "[FRAME] Failed to get session ID, closing connection\n";
        close(client_socket);
        return;
    }

    std::shared_ptr<ClientConnection> conn;

    {
        std::lock_guard<std::mutex> lock(clients_mutex);
        if (clients.find(session_id) == clients.end())
        {
            conn = std::make_shared<ClientConnection>();
            conn->client_id = session_id;

            // INITIALIZE DEFAULT CONFIG
            conn->config.output_index = 0;
            conn->config.fps = 30;
            conn->config.term_width = 80;
            conn->config.term_height = 24;
            conn->config.color_mode = 1; // 256 colors
            conn->config.renderer = 3;   // hybrid
            conn->config.keep_aspect_ratio = 1;
            conn->config.scale_factor = 1.0;
            conn->config.detail_level = 50; // mid detail
            conn->config.follow_focus = 0;

            clients[session_id] = conn;
            std::cerr << "[FRAME] Created new client session: " << session_id << "\n";
        }
        else
        {
            conn = clients[session_id];
            std::cerr << "[FRAME] Matched existing session: " << session_id << "\n";
        }
        conn->frame_socket = client_socket;
        conn->active = true;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Brief pause to ensure stability

    // Configure socket for performance
    int sndbuf = 16 * 1024 * 1024;
    setsockopt(client_socket, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));
    int nodelay = 1;
    setsockopt(client_socket, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

    // Create async send queue and thread
    auto send_queue = std::make_shared<SendQueue>();
    std::thread sender(send_thread, client_socket, send_queue);

    if (!conn->active || !running)
    {
        std::cerr << "[FRAME] Connection inactive, closing\n";
        goto cleanup;
    }

    // START FRAME LOOP
    {
        // Get reference to config for updates
        ClientConfig &config = conn->config;

        uint32_t output_index;
        if (config.follow_focus)
        {
            output_index = focus_tracker.focused_output_index.load();
            output_index = std::min(output_index, (uint32_t)(outputs.size() - 1));
            std::cerr << "[FRAME] Following focus to output " << output_index << "\n";
        }
        else
        {
            output_index = config.output_index;
            output_index = std::min(output_index, (uint32_t)(outputs.size() - 1));
        }

        struct RenderedFrame {
            std::vector<uint8_t> data;
            std::chrono::steady_clock::time_point timestamp;
        };

        std::queue<RenderedFrame> rendered_queue;
        std::mutex render_mutex;
        std::condition_variable render_cv;
        std::atomic<bool> renderer_running{true};

        // Rendering thread - runs independently
        auto render_thread_func = [&]() {
            std::cerr << "[RENDER] Thread started for session " << session_id << "\n";
            
            auto last_render = std::chrono::steady_clock::now();
            long render_delay = 1000000 / config.fps;
            
            while (running && conn->active && renderer_running) {
                auto render_start = std::chrono::steady_clock::now();
                
                // Update output index for focus following
                uint32_t current_output = output_index;
                if (config.follow_focus) {
                    current_output = focus_tracker.focused_output_index.load();
                    current_output = std::min(current_output, (uint32_t)(outputs.size() - 1));
                } else {
                    current_output = config.output_index;
                    current_output = std::min(current_output, (uint32_t)(outputs.size() - 1));
                }
                
                // Capture frame
                std::vector<uint8_t> local_frame;
                uint32_t width = 0, height = 0, stride = 0;
                
                {
                    std::lock_guard<std::mutex> lock(*output_mutexes[current_output]);
                    Capture &cap = output_captures[current_output];
                    
                    if (!cap.back_ready) {
                        std::this_thread::sleep_for(std::chrono::microseconds(500));
                        continue;
                    }
                    
                    local_frame = cap.back_buffer;
                    width = cap.width;
                    height = cap.height;
                    stride = cap.stride;
                }
                
                // Apply zoom if enabled
                std::vector<uint8_t> frame_to_render = local_frame;
                uint32_t render_width = width;
                uint32_t render_height = height;
                uint32_t render_stride = stride;
                
                if (conn->zoom.enabled) {
                    update_zoom_smooth_pan(conn->zoom);
                    
                    std::vector<uint8_t> zoomed_frame;
                    uint32_t zoomed_width, zoomed_height, zoomed_stride;
                    
                    apply_zoom_transform(
                        local_frame.data(), width, height, stride,
                        zoomed_frame, zoomed_width, zoomed_height, zoomed_stride,
                        conn->zoom);
                    
                    frame_to_render = zoomed_frame;
                    render_width = zoomed_width;
                    render_height = zoomed_height;
                    render_stride = zoomed_stride;
                }
                
                // Render frame
                ColorMode mode = static_cast<ColorMode>(config.color_mode);
                bool keep_aspect_ratio = config.keep_aspect_ratio != 0;
                
                std::string rendered;
                switch (config.renderer) {
                    case 0:
                        if (config.render_device == 1) {
                            rendered = render_braille_cuda_wrapper(
                                frame_to_render.data(), render_width, render_height, render_stride,
                                config.term_width, config.term_height,
                                mode, keep_aspect_ratio, config.scale_factor, 
                                config.detail_level, config.quality, config.rotation_angle);
                        } else {
                            rendered = render_braille(
                                frame_to_render.data(), render_width, render_height, render_stride,
                                config.term_width, config.term_height,
                                mode, keep_aspect_ratio, config.scale_factor, 
                                config.detail_level, config.quality, config.rotation_angle);
                        }
                        break;
                    case 1:
                        rendered = render_blocks(
                            frame_to_render.data(), render_width, render_height, render_stride,
                            config.term_width, config.term_height,
                            mode, keep_aspect_ratio, config.scale_factor, 
                            config.detail_level, config.quality, config.rotation_angle);
                        break;
                    case 2:
                        rendered = render_ascii(
                            frame_to_render.data(), render_width, render_height, render_stride,
                            config.term_width, config.term_height,
                            mode, keep_aspect_ratio, config.scale_factor, 
                            config.detail_level, config.quality, config.rotation_angle);
                        break;
                    case 3:
                    default:
                        if (config.render_device == 1) {
                            rendered = render_hybrid_cuda_wrapper(
                                frame_to_render.data(), render_width, render_height, render_stride,
                                config.term_width, config.term_height,
                                mode, keep_aspect_ratio, config.scale_factor, 
                                config.detail_level, config.quality, config.rotation_angle);
                        } else {
                            rendered = render_hybrid(
                                frame_to_render.data(), render_width, render_height, render_stride,
                                config.term_width, config.term_height,
                                mode, keep_aspect_ratio, config.scale_factor, 
                                config.detail_level, config.quality, config.rotation_angle);
                        }
                        break;
                }
                
                if (rendered.empty()) {
                    std::cerr << "[RENDER] Warning: Empty rendered frame\n";
                    continue;
                }
                
                // Build complete message with screen info
                std::vector<uint8_t> info_msg;
                MessageType info_type = MessageType::SCREEN_INFO;
                ScreenInfo info{width, height};
                
                const uint8_t *type_bytes = reinterpret_cast<const uint8_t *>(&info_type);
                info_msg.insert(info_msg.end(), type_bytes, type_bytes + sizeof(info_type));
                const uint8_t *info_bytes = reinterpret_cast<const uint8_t *>(&info);
                info_msg.insert(info_msg.end(), info_bytes, info_bytes + sizeof(info));
                
                // Build frame message
                std::vector<uint8_t> frame_msg;
                
                if (config.compress) {
                    // Build uncompressed frame first
                    std::vector<uint8_t> uncompressed;
                    MessageType msg_type = MessageType::RENDERED_FRAME;
                    type_bytes = reinterpret_cast<const uint8_t *>(&msg_type);
                    uncompressed.insert(uncompressed.end(), type_bytes, type_bytes + sizeof(msg_type));
                    
                    RenderedFrameHeader header;
                    header.data_size = rendered.size();
                    const uint8_t *header_bytes = reinterpret_cast<const uint8_t *>(&header);
                    uncompressed.insert(uncompressed.end(), header_bytes, header_bytes + sizeof(header));
                    
                    const uint8_t *data_bytes = reinterpret_cast<const uint8_t *>(rendered.data());
                    uncompressed.insert(uncompressed.end(), data_bytes, data_bytes + rendered.size());
                    
                    std::vector<uint8_t> compressed = compress_frame(uncompressed, config.compression_level);
                    
                    MessageType comp_type = MessageType::COMPRESSED_FRAME;
                    type_bytes = reinterpret_cast<const uint8_t *>(&comp_type);
                    frame_msg.insert(frame_msg.end(), type_bytes, type_bytes + sizeof(comp_type));
                    
                    CompressedFrameHeader comp_header;
                    comp_header.compressed_size = compressed.size();
                    comp_header.uncompressed_size = uncompressed.size();
                    const uint8_t *comp_bytes = reinterpret_cast<const uint8_t *>(&comp_header);
                    frame_msg.insert(frame_msg.end(), comp_bytes, comp_bytes + sizeof(comp_header));
                    frame_msg.insert(frame_msg.end(), compressed.begin(), compressed.end());
                } else {
                    MessageType msg_type = MessageType::RENDERED_FRAME;
                    type_bytes = reinterpret_cast<const uint8_t *>(&msg_type);
                    frame_msg.insert(frame_msg.end(), type_bytes, type_bytes + sizeof(msg_type));
                    
                    RenderedFrameHeader header;
                    header.data_size = rendered.size();
                    const uint8_t *header_bytes = reinterpret_cast<const uint8_t *>(&header);
                    frame_msg.insert(frame_msg.end(), header_bytes, header_bytes + sizeof(header));
                    
                    const uint8_t *data_bytes = reinterpret_cast<const uint8_t *>(rendered.data());
                    frame_msg.insert(frame_msg.end(), data_bytes, data_bytes + rendered.size());
                }
                
                // Combine messages
                std::vector<uint8_t> complete_msg;
                complete_msg.reserve(info_msg.size() + frame_msg.size());
                complete_msg.insert(complete_msg.end(), info_msg.begin(), info_msg.end());
                complete_msg.insert(complete_msg.end(), frame_msg.begin(), frame_msg.end());
                
                // Add to rendered queue
                {
                    std::lock_guard<std::mutex> lock(render_mutex);
                    
                    // Drop old frames if queue is full (keep max 2 frames buffered)
                    while (rendered_queue.size() >= 2) {
                        rendered_queue.pop();
                        frames_dropped++;
                    }
                    
                    RenderedFrame rf;
                    rf.data = std::move(complete_msg);
                    rf.timestamp = std::chrono::steady_clock::now();
                    rendered_queue.push(std::move(rf));
                    render_cv.notify_one();
                }
                
                // Frame pacing
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                    now - last_render).count();
                
                if (elapsed < render_delay) {
                    usleep(render_delay - elapsed);
                }
                
                last_render = std::chrono::steady_clock::now();
            }
            
            std::cerr << "[RENDER] Thread stopped for session " << session_id << "\n";
        };

        // Start rendering thread
        std::thread render_worker(render_thread_func);

        // Send thread - just pulls from queue and sends
        std::cerr << "[FRAME] Starting send loop for session " << session_id << "\n";

        while (running && conn->active && send_queue->running) {
            RenderedFrame frame;
            
            {
                std::unique_lock<std::mutex> lock(render_mutex);
                
                // Wait for rendered frame
                render_cv.wait_for(lock, std::chrono::milliseconds(100), [&] {
                    return !rendered_queue.empty() || !running || !conn->active;
                });
                
                if (!running || !conn->active) break;
                
                if (rendered_queue.empty()) continue;
                
                frame = std::move(rendered_queue.front());
                rendered_queue.pop();
            }
            
            // Queue for async sending
            {
                std::lock_guard<std::mutex> lock(send_queue->mutex);
                if (send_queue->frame_ready) {
                    send_queue->dropped_frames++;
                }
                send_queue->latest_frame = std::move(frame.data);
                send_queue->frame_ready = true;
                send_queue->cv.notify_one();
            }
            
            total_frames_sent++;
        }

        // Cleanup
        renderer_running = false;
        render_cv.notify_one();
        render_worker.join();
    }

cleanup:
    send_queue->running = false;
    send_queue->cv.notify_one();
    sender.join();

    conn->active = false;
    close(client_socket);
    std::cerr << "[FRAME] Client handler exiting for session " << session_id << "\n";
}

static void reset_modifier_state()
{
    server_shift_pressed = false;
    server_ctrl_pressed = false;
    server_alt_pressed = false;
    server_super_pressed = false;
    server_altgr_pressed = false;
    server_capslock_pressed = false;
    server_numlock_pressed = false;

    // Send modifier reset to compositor
    if (virtual_keyboard)
    {
        zwp_virtual_keyboard_v1_modifiers(virtual_keyboard, 0, 0, 0, 0);
        wl_display_flush(display);
        std::cerr << "[INPUT] Modifier state reset\n";
    }
}

static void handle_config_client(int client_socket, sockaddr_in client_addr)
{
    std::cerr << "[CONFIG] Client connected from " << inet_ntoa(client_addr.sin_addr) << "\n";

    // Receive session ID
    std::string session_id;
    if (!receive_session_id(client_socket, session_id))
    {
        std::cerr << "[CONFIG] Failed to get session ID, closing connection\n";
        close(client_socket);
        return;
    }

    std::shared_ptr<ClientConnection> conn;

    {
        std::lock_guard<std::mutex> lock(clients_mutex);
        if (clients.find(session_id) == clients.end())
        {
            conn = std::make_shared<ClientConnection>();
            conn->client_id = session_id;

            // Initialize default config
            conn->config.output_index = 0;
            conn->config.fps = 30;
            conn->config.term_width = 80;
            conn->config.term_height = 24;
            conn->config.color_mode = 1;
            conn->config.renderer = 3;
            conn->config.keep_aspect_ratio = 1;
            conn->config.scale_factor = 1.0;
            conn->config.compress = 0;
            conn->config.compression_level = 0;
            conn->config.follow_focus = 0;

            clients[session_id] = conn;
            std::cerr << "[CONFIG] Created new client session: " << session_id << "\n";
        }
        else
        {
            conn = clients[session_id];
            std::cerr << "[CONFIG] Matched existing session: " << session_id << "\n";
        }
        conn->config_socket = client_socket;
        conn->active = true;
    }

    // Configure socket
    int nodelay = 1;
    setsockopt(client_socket, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

    // Use poll() for event-driven I/O
    struct pollfd pfd;
    pfd.fd = client_socket;
    pfd.events = POLLIN;

    while (running && conn->active)
    {
        int ret = poll(&pfd, 1, 100); // 100ms timeout

        if (ret < 0)
        {
            if (errno == EINTR)
                continue;
            std::cerr << "[CONFIG] Poll error: " << strerror(errno) << "\n";
            break;
        }

        if (ret == 0)
            continue;

        if (pfd.revents & POLLIN)
        {
            MessageType type;
            ssize_t n = recv(client_socket, &type, sizeof(type), MSG_DONTWAIT);

            if (n == sizeof(type))
            {
                if (type == MessageType::CLIENT_CONFIG)
                {
                    ClientConfig new_config;
                    n = recv(client_socket, &new_config, sizeof(new_config), 0);
                    if (n == sizeof(new_config))
                    {
                        std::lock_guard<std::mutex> lock(clients_mutex);
                        conn->config = new_config;

                        std::cerr << "[DEBUG] Received config: follow_focus=" << (int)new_config.follow_focus 
                                  << " output_index=" << new_config.output_index
                                  << " renderer=" << (int)new_config.renderer
                                  << " detail=" << (int)new_config.detail_level
                                  << " quality=" << (int)new_config.quality << "\n";

                        bool old_follow = focus_tracker.follow_focus.load();
                        bool new_follow = new_config.follow_focus != 0;

                        if (new_follow != old_follow)
                        {
                            focus_tracker.follow_focus = new_follow;
                            std::cerr << "[FOCUS] Focus following "
                                      << (new_follow ? "ENABLED" : "DISABLED") << "\n";

                            if (new_follow)
                            {
                                update_focus_tracking();
                            }
                        }

                        std::cerr << "[CONFIG] Update for session " << session_id << "\n";
                    }
                }
                // ZOOM: Handle zoom config
                else if (type == MessageType::ZOOM_CONFIG)
                {
                    ZoomConfig zoom_config;
                    n = recv(client_socket, &zoom_config, sizeof(zoom_config), 0);
                    if (n == sizeof(zoom_config))
                    {
                        std::lock_guard<std::mutex> lock(clients_mutex);
                        std::lock_guard<std::mutex> zoom_lock(conn->zoom.mutex);

                        conn->zoom.enabled = zoom_config.enabled != 0;
                        conn->zoom.follow_mouse = zoom_config.follow_mouse != 0;
                        conn->zoom.zoom_level = std::clamp(zoom_config.zoom_level, 1.0, 10.0);
                        conn->zoom.view_width = zoom_config.view_width;
                        conn->zoom.view_height = zoom_config.view_height;
                        conn->zoom.smooth_pan = zoom_config.smooth_pan != 0;
                        conn->zoom.pan_speed = zoom_config.pan_speed;

                        // Update target position
                        if (conn->zoom.follow_mouse)
                        {
                            conn->zoom.target_center_x = zoom_config.center_x;
                            conn->zoom.target_center_y = zoom_config.center_y;

                            // If not using smooth pan, snap immediately
                            if (!conn->zoom.smooth_pan)
                            {
                                conn->zoom.center_x = zoom_config.center_x;
                                conn->zoom.center_y = zoom_config.center_y;
                            }
                        }
                        else
                        {
                            // Static zoom - use provided center
                            conn->zoom.center_x = zoom_config.center_x;
                            conn->zoom.center_y = zoom_config.center_y;
                            conn->zoom.target_center_x = zoom_config.center_x;
                            conn->zoom.target_center_y = zoom_config.center_y;
                        }

                        std::cerr << "[ZOOM] Config update: "
                                  << (conn->zoom.enabled ? "ENABLED" : "DISABLED")
                                  << " level=" << conn->zoom.zoom_level << "x"
                                  << " viewport=" << conn->zoom.view_width << "x" << conn->zoom.view_height
                                  << " center=(" << conn->zoom.center_x << "," << conn->zoom.center_y << ")"
                                  << " follow=" << (conn->zoom.follow_mouse ? "YES" : "NO") << "\n";
                    }
                }
                else if (n == 0)
                {
                    break;
                }
            }
        }
    }

    conn->active = false;
    close(client_socket);
    std::cerr << "[CONFIG] Client disconnected for session " << session_id << "\n";
}

// NEW: Accept config connections
static void accept_config_thread(int server_socket)
{
    std::cerr << "[CONFIG] Accept thread started\n";

    while (running)
    {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);

        int client_socket = accept(server_socket, (sockaddr *)&client_addr, &client_len);
        if (client_socket < 0)
        {
            if (running)
            {
                std::cerr << "[CONFIG] Failed to accept client\n";
            }
            continue;
        }

        std::cerr << "[CONFIG] Client connecting from " << inet_ntoa(client_addr.sin_addr) << "\n";
        std::thread(handle_config_client, client_socket, client_addr).detach();
    }

    std::cerr << "[CONFIG] Accept thread stopped\n";
}

// UPDATED: Remove config handling from input client
static void handle_input_client(int client_socket, sockaddr_in client_addr)
{
    std::cerr << "[INPUT] Client connected from " << inet_ntoa(client_addr.sin_addr) << "\n";

    std::string session_id;
    if (!receive_session_id(client_socket, session_id))
    {
        std::cerr << "[INPUT] Failed to get session ID, closing connection\n";
        close(client_socket);
        return;
    }

    std::shared_ptr<ClientConnection> conn;

    {
        std::lock_guard<std::mutex> lock(clients_mutex);
        if (clients.find(session_id) == clients.end())
        {
            conn = std::make_shared<ClientConnection>();
            conn->client_id = session_id;
            clients[session_id] = conn;
            std::cerr << "[INPUT] Created new client session: " << session_id << "\n";
        }
        else
        {
            conn = clients[session_id];
            std::cerr << "[INPUT] Matched existing session: " << session_id << "\n";
        }
        conn->input_socket = client_socket;
        conn->active = true;
    }

    int nodelay = 1;
    setsockopt(client_socket, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

    struct pollfd pfd;
    pfd.fd = client_socket;
    pfd.events = POLLIN;

    while (running && conn->active)
    {
        int ret = poll(&pfd, 1, 1);

        if (ret < 0)
        {
            if (errno == EINTR)
                continue;
            std::cerr << "[INPUT] Poll error: " << strerror(errno) << "\n";
            break;
        }

        if (ret == 0)
            continue;

        if (pfd.revents & POLLIN)
        {
            MessageType type;
            ssize_t n = recv(client_socket, &type, sizeof(type), MSG_DONTWAIT);

            if (n == sizeof(type))
            {
                // Only handle input events, not config
                InputEvent evt;
                evt.type = type;
                bool valid = false;

                switch (type)
                {
                case MessageType::KEY_EVENT:
                    if (recv(client_socket, &evt.key, sizeof(evt.key), 0) == sizeof(evt.key))
                        valid = true;
                    break;
                case MessageType::MOUSE_MOVE:
                    if (recv(client_socket, &evt.mouse_move, sizeof(evt.mouse_move), 0) == sizeof(evt.mouse_move))
                        valid = true;
                    break;
                case MessageType::MOUSE_BUTTON:
                    if (recv(client_socket, &evt.mouse_button, sizeof(evt.mouse_button), 0) == sizeof(evt.mouse_button))
                        valid = true;
                    break;
                case MessageType::MOUSE_SCROLL:
                    if (recv(client_socket, &evt.mouse_scroll, sizeof(evt.mouse_scroll), 0) == sizeof(evt.mouse_scroll))
                        valid = true;
                    break;
                default:
                    std::cerr << "[INPUT] Unexpected message type: " << (int)type << "\n";
                    break;
                }

                if (valid)
                {
                    std::lock_guard<std::mutex> lock(input_mutex);
                    evt.client_id = session_id;
                    input_queue.push(evt);
                    input_cv.notify_one();
                }
            }
            else if (n == 0)
            {
                break;
            }
            else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
            {
                break;
            }
        }

        if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))
        {
            break;
        }
    }

    reset_modifier_state();
    conn->active = false;
    close(client_socket);
    std::cerr << "[INPUT] Client disconnected for session " << session_id << "\n";
}

static void accept_frame_thread(int server_socket)
{
    std::cerr << "[FRAME] Accept thread started\n";

    while (running)
    {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);

        int client_socket = accept(server_socket, (sockaddr *)&client_addr, &client_len);
        if (client_socket < 0)
        {
            if (running)
            {
                std::cerr << "[FRAME] Failed to accept client\n";
            }
            continue;
        }

        std::cerr << "[FRAME] Client connecting from " << inet_ntoa(client_addr.sin_addr) << "\n";
        std::thread(handle_frame_client, client_socket, client_addr).detach();
    }

    std::cerr << "[FRAME] Accept thread stopped\n";
}

static void accept_input_thread(int server_socket)
{
    std::cerr << "[INPUT] Accept thread started\n";

    while (running)
    {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);

        int client_socket = accept(server_socket, (sockaddr *)&client_addr, &client_len);
        if (client_socket < 0)
        {
            if (running)
            {
                std::cerr << "[INPUT] Failed to accept client\n";
            }
            continue;
        }

        std::cerr << "[INPUT] Client connecting from " << inet_ntoa(client_addr.sin_addr) << "\n";
        std::thread(handle_input_client, client_socket, client_addr).detach();
    }

    std::cerr << "[INPUT] Accept thread stopped\n";
}

static void wayland_dispatch_thread()
{
    while (running)
    {
        if (wl_display_dispatch(display) == -1)
        {
            std::cerr << "Display dispatch failed\n";
        }
    }
}

static void shutdown_handler(int signum)
{
    static bool first_interrupt = true;

    if (first_interrupt)
    {
        first_interrupt = false;
        std::cerr << "\n[SHUTDOWN] Interrupt received, stopping...\n";
        running = false;

        // CRITICAL: Close all server sockets immediately to unblock accept()
        for (int sock : all_server_sockets)
        {
            if (sock >= 0)
            {
                shutdown(sock, SHUT_RDWR); // Force immediate shutdown
                close(sock);
            }
        }

        // Wake up all condition variables
        input_cv.notify_all();
        for (auto &cv : frame_ready_cvs)
        {
            cv->notify_all();
        }

        // Force close all client connections
        {
            std::lock_guard<std::mutex> lock(clients_mutex);
            for (auto &[id, conn] : clients)
            {
                conn->active = false;
                if (conn->frame_socket >= 0)
                {
                    shutdown(conn->frame_socket, SHUT_RDWR);
                }
                if (conn->input_socket >= 0)
                {
                    shutdown(conn->input_socket, SHUT_RDWR);
                }
                if (conn->config_socket >= 0)
                {
                    shutdown(conn->config_socket, SHUT_RDWR);
                }
            }
        }

        // Stop audio/microphone
        if (feature_audio)
        {
            audio_capture.running = false;
        }
        if (feature_microphone)
        {
            microphone_virtual_source.running = false;
        }
    }
    else
    {
        // Second Ctrl+C = force exit
        std::cerr << "\n[SHUTDOWN] Force exit!\n";
        _exit(1);
    }
}

int main(int argc, char **argv)
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    argparse::ArgumentParser program("waytermirror_server");
    program.add_argument("-P", "--port")
        .default_value(9999)
        .scan<'i', int>()
        .help("Server port");

    program.add_argument("-F", "--capture-fps")
        .default_value(30)
        .scan<'i', int>()
        .help("Default capture FPS");

    program.add_argument("-n", "--no-video")
        .default_value(false)
        .implicit_value(true)
        .help("Disable video");

    program.add_argument("-A", "--no-audio")
        .default_value(false)
        .implicit_value(true)
        .help("Disable audio");

    program.add_argument("-N", "--no-input")
        .default_value(false)
        .implicit_value(true)
        .help("Disable input forwarding");

    program.add_argument("-m", "--no-microphone")
        .default_value(false)
        .implicit_value(true)
        .help("Disable microphone input forwarding");

    program.add_argument("-C", "--compositor")
        .default_value(std::string("auto"))
        .help("Compositor type: auto, hyprland, sway, kde, gnome, generic");

    program.add_argument("-B", "--capture-backend")
        .default_value(std::string("auto"))
        .help("Capture backend: auto, wlr, pipewire");

    program.add_argument("-I", "--input-backend")
        .default_value(std::string("auto"))
        .help("Input backend: auto, virtual, uinput");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (...)
    {
        std::cerr << program;
        return 1;
    }

    std::string backend_str = program.get<std::string>("--capture-backend");
    if (backend_str == "pipewire") {
        capture_backend = PIPEWIRE;
    } else if (backend_str == "wlr") {
        capture_backend = WLR_SCREENCOPY;
    } else {
        capture_backend = AUTO_CAPTURE;
    }
    
    std::string input_backend_str = program.get<std::string>("--input-backend");
    VirtualInputManager::Backend preferred_input_backend = VirtualInputManager::AUTO;
    if (input_backend_str == "virtual") {
        preferred_input_backend = VirtualInputManager::WLR_PROTOCOLS;
    } else if (input_backend_str == "uinput") {
        preferred_input_backend = VirtualInputManager::UINPUT;
    } else {
        preferred_input_backend = VirtualInputManager::AUTO;
    }
    
    std::string compositor_override = program.get<std::string>("--compositor");
    compositor_type = detect_compositor(compositor_override);
    std::cerr << "[INIT] Compositor: " << compositor_type << "\n";

    if (compositor_type == "hyprland" || compositor_type == "sway")
    {
        std::cerr << "[INIT] Will use " << compositor_type << " CLI tools for focus detection\n";
    }
    else if (compositor_type == "generic")
    {
        std::cerr << "[INIT] Will use toplevel protocol for focus detection (may be unreliable)\n";
    }
    else
    {
        std::cerr << "[INIT] Will try " << compositor_type << " CLI tools, fallback to toplevel protocol\n";
    }

    int port = program.get<int>("--port");
    int capture_fps = program.get<int>("--capture-fps");

    feature_video = !program.get<bool>("--no-video");
    feature_audio = !program.get<bool>("--no-audio");
    feature_input = !program.get<bool>("--no-input");
    feature_microphone = !program.get<bool>("--no-microphone");

    std::cerr << "=== Features ===\n";
    std::cerr << "Video: " << (feature_video ? "ON" : "OFF") << "\n";
    std::cerr << "Audio: " << (feature_audio ? "ON" : "OFF") << "\n";
    std::cerr << "Input: " << (feature_input ? "ON" : "OFF") << "\n";
    std::cerr << "Microphone: " << (feature_microphone ? "ON" : "OFF") << "\n";
    std::cerr << "================\n\n";

    // Connect to Wayland (needed for video OR input)
    if (feature_video || feature_input)
    {
        display = wl_display_connect(nullptr);
        if (!display)
        {
            std::cerr << "Failed to connect to Wayland\n";
            return 1;
        }

        registry = wl_display_get_registry(display);
        wl_registry_add_listener(registry, &registry_listener, nullptr);
        wl_display_roundtrip(display);

        if (toplevel_manager)
        {
            zwlr_foreign_toplevel_manager_v1_add_listener(toplevel_manager, &manager_listener, nullptr);
            std::cerr << "[FOCUS] Added toplevel manager listener\n";

            // This roundtrip will trigger manager_handle_toplevel for existing windows
            // and those will get listeners which will receive their initial state
            wl_display_roundtrip(display);

            std::cerr << "[FOCUS] Focus tracking enabled with "
                      << focus_tracker.toplevels.size() << " existing windows\n";
        }

        if (feature_video && (!shm || outputs.empty() || !manager))
        {
            std::cerr << "Failed to get required Wayland globals for video\n";
            return 1;
        }

        if (feature_video)
        {
            std::cerr << "Found " << outputs.size() << " output(s)\n";

            // Initialize capture structures
            output_captures.resize(outputs.size());
            for (size_t i = 0; i < outputs.size(); i++)
            {
                output_mutexes.push_back(std::make_unique<std::mutex>());
                frame_ready_cvs.push_back(std::make_unique<std::condition_variable>());
            }
            output_ready.resize(outputs.size(), false);
        }

        // Initialize virtual input devices
        if (feature_input)
        {
            if (seat && pointer_manager)
            {
                virtual_pointer = zwlr_virtual_pointer_manager_v1_create_virtual_pointer(pointer_manager, seat);
                std::cerr << "Virtual pointer initialized\n";
            }

            if (seat && keyboard_manager)
            {
                virtual_keyboard = zwp_virtual_keyboard_manager_v1_create_virtual_keyboard(keyboard_manager, seat);
                setup_virtual_keyboard_keymap();
                std::cerr << "Virtual keyboard initialized\n";
            }

            // Diagnostic checks
            std::cerr << "\n=== Virtual Input Diagnostics ===\n";
            std::cerr << "Seat available: " << (seat ? "YES" : "NO") << "\n";
            std::cerr << "Pointer manager available: " << (pointer_manager ? "YES" : "NO") << "\n";
            std::cerr << "Keyboard manager available: " << (keyboard_manager ? "YES" : "NO") << "\n";
            std::cerr << "Virtual pointer created: " << (virtual_pointer ? "YES" : "NO") << "\n";
            std::cerr << "Virtual keyboard created: " << (virtual_keyboard ? "YES" : "NO") << "\n";

            if (!virtual_pointer || !virtual_keyboard)
            {
                std::cerr << "\nWARNING: Virtual input devices not available!\n";
                std::cerr << "Your compositor may not support:\n";
                std::cerr << "  - zwlr_virtual_pointer_manager_v1\n";
                std::cerr << "  - zwp_virtual_keyboard_manager_v1\n";
                std::cerr << "\nSupported compositors: Hyprland, Sway, etc.\n";
                std::cerr << "NOT supported: GNOME, KDE Plasma (without plugins)\n\n";
            }
            std::cerr << "================================\n\n";
        }
    }

    if (capture_backend == AUTO_CAPTURE) {
        capture_backend = detect_capture_backend();
    }

    std::thread focus_updater;
    if (toplevel_manager)
    {
        focus_updater = std::thread(focus_update_thread);
    }

    // Initialize audio capture
    std::thread audio_acceptor;
    if (feature_audio)
    {
        if (!init_audio_capture())
        {
            std::cerr << "Warning: Audio init failed\n";
            feature_audio = false;
        }
        else
        {
            audio_server_socket = socket(AF_INET, SOCK_STREAM, 0);
            all_server_sockets.push_back(audio_server_socket);
            int opt = 1;
            setsockopt(audio_server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

            sockaddr_in audio_addr{};
            audio_addr.sin_family = AF_INET;
            audio_addr.sin_addr.s_addr = INADDR_ANY;
            audio_addr.sin_port = htons(port + 2);

            if (bind(audio_server_socket, (sockaddr *)&audio_addr, sizeof(audio_addr)) < 0)
            {
                std::cerr << "Failed to bind audio socket\n";
                cleanup_audio_capture();
                feature_audio = false;
            }
            else
            {
                listen(audio_server_socket, 10);
                std::cerr << "Audio server listening on port " << (port + 2) << "\n";
                audio_acceptor = std::thread(accept_audio_thread, audio_server_socket);
            }
        }
    }

    // Setup microphone input server socket (port + 4)
    std::thread microphone_acceptor;
    if (feature_microphone)
    {
        microphone_server_socket = socket(AF_INET, SOCK_STREAM, 0);
        all_server_sockets.push_back(microphone_server_socket);
        if (microphone_server_socket < 0)
        {
            std::cerr << "Warning: Failed to create microphone socket\n";
            feature_microphone = false;
        }
        else
        {
            int opt = 1;
            setsockopt(microphone_server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

            sockaddr_in microphone_addr{};
            microphone_addr.sin_family = AF_INET;
            microphone_addr.sin_addr.s_addr = INADDR_ANY;
            microphone_addr.sin_port = htons(port + 4);

            if (bind(microphone_server_socket, (sockaddr *)&microphone_addr, sizeof(microphone_addr)) < 0)
            {
                std::cerr << "Failed to bind microphone socket on port " << (port + 4) << "\n";
                close(microphone_server_socket);
                microphone_server_socket = -1;
                feature_microphone = false;
            }
            else
            {
                if (listen(microphone_server_socket, 10) < 0)
                {
                    std::cerr << "Failed to listen on microphone socket\n";
                    close(microphone_server_socket);
                    microphone_server_socket = -1;
                    feature_microphone = false;
                }
                else
                {
                    std::cerr << "Microphone server listening on port " << (port + 4) << "\n";
                    microphone_acceptor = std::thread(accept_microphone_thread, microphone_server_socket);
                }
            }
        }
    }

    // Setup frame server socket
    if (feature_video)
    {
        frame_server_socket = socket(AF_INET, SOCK_STREAM, 0);
        all_server_sockets.push_back(frame_server_socket);
        if (frame_server_socket < 0)
        {
            std::cerr << "Failed to create frame socket\n";
            return 1;
        }

        int opt = 1;
        setsockopt(frame_server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in frame_addr{};
        frame_addr.sin_family = AF_INET;
        frame_addr.sin_addr.s_addr = INADDR_ANY;
        frame_addr.sin_port = htons(port);

        if (bind(frame_server_socket, (sockaddr *)&frame_addr, sizeof(frame_addr)) < 0)
        {
            std::cerr << "Failed to bind frame socket\n";
            return 1;
        }

        if (listen(frame_server_socket, 10) < 0)
        {
            std::cerr << "Failed to listen on frame socket\n";
            return 1;
        }

        std::cerr << "Frame server listening on port " << port << "\n";
    }

    // Setup input server socket
    if (feature_input)
    {
        input_server_socket = socket(AF_INET, SOCK_STREAM, 0);
        all_server_sockets.push_back(input_server_socket);
        if (input_server_socket < 0)
        {
            std::cerr << "Failed to create input socket\n";
            return 1;
        }

        int opt = 1;
        setsockopt(input_server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in input_addr{};
        input_addr.sin_family = AF_INET;
        input_addr.sin_addr.s_addr = INADDR_ANY;
        input_addr.sin_port = htons(port + 1);

        if (bind(input_server_socket, (sockaddr *)&input_addr, sizeof(input_addr)) < 0)
        {
            std::cerr << "Failed to bind input socket\n";
            return 1;
        }

        if (listen(input_server_socket, 10) < 0)
        {
            std::cerr << "Failed to listen on input socket\n";
            return 1;
        }

        std::cerr << "Input server listening on port " << (port + 1) << "\n";
    }

    // Initialize capture based on backend:
    if (feature_video) {
        if (capture_backend == PIPEWIRE) {
            std::cerr << "Initializing PipeWire capture for all outputs\n";
            
            pipewire_captures.resize(outputs.size());
            
            // Initialize each output with its own portal session
            for (size_t i = 0; i < outputs.size(); i++) {
                pipewire_captures[i] = std::make_unique<PipeWireCapture>();
                if (!pipewire_captures[i]->init(i)) {
                    std::cerr << "Failed to initialize PipeWire capture for output " << i << "\n";
                    return 1;
                }
            }
            
            for (size_t i = 0; i < outputs.size(); i++) {
                output_mutexes.push_back(std::make_unique<std::mutex>());
                frame_ready_cvs.push_back(std::make_unique<std::condition_variable>());
            }
            output_ready.resize(outputs.size(), false);
            output_captures.resize(outputs.size());
        } else {
            // Original wlr-screencopy initialization
            std::cerr << "Found " << outputs.size() << " output(s)\n";
    
            output_captures.resize(outputs.size());
            for (size_t i = 0; i < outputs.size(); i++) {
                output_mutexes.push_back(std::make_unique<std::mutex>());
                frame_ready_cvs.push_back(std::make_unique<std::condition_variable>());
            }
            output_ready.resize(outputs.size(), false);
        }
    }
    
    // Initialize virtual input after checking protocols:
    if (feature_input) {
        // Check what's available
        bool has_wlr_input = (seat && pointer_manager && keyboard_manager);
        
        VirtualInputManager::Backend input_backend;
        if (preferred_input_backend == VirtualInputManager::AUTO) {
            // Auto-detect best available
            if (has_wlr_input) {
                input_backend = VirtualInputManager::WLR_PROTOCOLS;
                std::cerr << "[INPUT] Auto-selected WLR virtual input protocols\n";
            } else if (uinput_available()) {
                input_backend = VirtualInputManager::UINPUT;
                std::cerr << "[INPUT] Auto-selected uinput backend\n";
            } else {
                std::cerr << "[INPUT] No virtual input backend available!\n";
                feature_input = false;
            }
        } else if (preferred_input_backend == VirtualInputManager::WLR_PROTOCOLS) {
            // User explicitly requested WLR protocols
            if (has_wlr_input) {
                input_backend = VirtualInputManager::WLR_PROTOCOLS;
                std::cerr << "[INPUT] Using WLR virtual input protocols (manual)\n";
            } else {
                std::cerr << "[INPUT] WLR protocols not available!\n";
                feature_input = false;
            }
        } else if (preferred_input_backend == VirtualInputManager::UINPUT) {
            // User explicitly requested uinput
            if (uinput_available()) {
                input_backend = VirtualInputManager::UINPUT;
                std::cerr << "[INPUT] Using uinput backend (manual)\n";
            } else {
                std::cerr << "[INPUT] uinput not available!\n";
                feature_input = false;
            }
        }
        
        // Initialize the selected backend
        if (feature_input && input_backend == VirtualInputManager::UINPUT) {
            if (!virtual_input_mgr.init(input_backend)) {
                std::cerr << "[INPUT] Failed to initialize uinput backend\n";
                feature_input = false;
            }
        } else if (feature_input && input_backend == VirtualInputManager::WLR_PROTOCOLS) {
            // Keep using existing virtual_pointer and virtual_keyboard
        }
    }
    
    // Setup config server socket (ALWAYS enabled)
    config_server_socket = socket(AF_INET, SOCK_STREAM, 0);
    all_server_sockets.push_back(config_server_socket);
    if (config_server_socket < 0)
    {
        std::cerr << "Failed to create config socket\n";
        return 1;
    }

    int opt = 1;
    setsockopt(config_server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in config_addr{};
    config_addr.sin_family = AF_INET;
    config_addr.sin_addr.s_addr = INADDR_ANY;
    config_addr.sin_port = htons(port + 3); // Port+3 for config

    if (bind(config_server_socket, (sockaddr *)&config_addr, sizeof(config_addr)) < 0)
    {
        std::cerr << "Failed to bind config socket\n";
        return 1;
    }

    if (listen(config_server_socket, 10) < 0)
    {
        std::cerr << "Failed to listen on config socket\n";
        return 1;
    }

    std::cerr << "Config server listening on port " << (port + 3) << "\n";

    // Start config acceptor thread
    std::thread config_acceptor(accept_config_thread, config_server_socket);

    // Start threads based on enabled features
    std::thread input_processor;
    if (feature_input)
    {
        input_processor = std::thread(input_thread);
    }

    std::thread wayland_dispatcher;
    if (feature_video || feature_input)
    {
        wayland_dispatcher = std::thread(wayland_dispatch_thread);
    }

    std::vector<std::thread> capture_threads;
    if (feature_video)
    {
        for (size_t i = 0; i < outputs.size(); i++)
        {
            capture_threads.emplace_back(capture_thread, i, capture_fps);
        }
    }

    std::thread frame_acceptor;
    if (feature_video)
    {
        frame_acceptor = std::thread(accept_frame_thread, frame_server_socket);
    }

    std::thread input_acceptor;
    if (feature_input)
    {
        input_acceptor = std::thread(accept_input_thread, input_server_socket);
    }

    // Wait for interrupt
    std::cerr << "\n=== Waytermirror Server Started ===\n";
    std::cerr << "Listening on port " << port << "\n";
    if (feature_video)
    {
        std::cerr << "  Video (frames): port " << port << "\n";
    }
    if (feature_input)
    {
        std::cerr << "  Input: port " << (port + 1) << "\n";
    }
    if (feature_audio)
    {
        std::cerr << "  Audio (system out): port " << (port + 2) << "\n";
    }
    std::cerr << "  Config: port " << (port + 3) << "\n";
    if (feature_microphone)
    {
        std::cerr << "  Microphone (client in): port " << (port + 4) << "\n";
    }
    std::cerr << "Press Ctrl+C to stop...\n";
    std::cerr << "======================================\n\n";

    signal(SIGINT, [](int)
           { shutdown_handler(SIGINT); });
    signal(SIGTERM, [](int)
           { shutdown_handler(SIGTERM); });

    while (running)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    running = false;
    std::cerr << "\nShutting down...\n";

    // Join threads in reverse order
    if (feature_microphone && microphone_acceptor.joinable())
    {
        microphone_acceptor.join();
    }

    if (feature_audio && audio_acceptor.joinable())
    {
        audio_acceptor.join();
    }

    if (feature_video && frame_acceptor.joinable())
    {
        frame_acceptor.join();
    }

    if (feature_input && input_acceptor.joinable())
    {
        input_acceptor.join();
    }

    if (config_acceptor.joinable())
    {
        config_acceptor.join();
    }

    if (focus_updater.joinable())
    {
        focus_updater.join();
    }

    // Cleanup
    if (feature_input)
    {
        input_cv.notify_all();
        if (input_processor.joinable())
        {
            input_processor.join();
        }
    }

    if ((feature_video || feature_input) && wayland_dispatcher.joinable())
    {
        wayland_dispatcher.join();
    }

    for (auto &t : capture_threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    // Close sockets
    if (frame_server_socket >= 0)
        close(frame_server_socket);
    if (input_server_socket >= 0)
        close(input_server_socket);
    if (audio_server_socket >= 0)
        close(audio_server_socket);
    if (config_server_socket >= 0)
        close(config_server_socket);
    if (microphone_server_socket >= 0)
        close(microphone_server_socket);

    // Cleanup resources
    if (feature_video)
    {
        for (Capture &cap : output_captures)
        {
            if (cap.front_buffer)
            {
                wl_buffer_destroy(cap.front_buffer);
            }
        }
    }

    if (feature_audio)
    {
        cleanup_audio_capture();
    }

    if (feature_microphone)
    {
        cleanup_microphone_virtual_source();
    }

    if (feature_video || feature_input)
    {
        wl_display_disconnect(display);
    }

    if (feature_input) {
        virtual_input_mgr.cleanup();
    }
    
    if (capture_backend == PIPEWIRE) {
        for (auto &cap : pipewire_captures) {
            if (cap) cap->cleanup();
        }
    }

    std::cerr << "Shutdown complete.\n";
    return 0;
}
