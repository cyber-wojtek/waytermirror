#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <cstring>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <argparse/argparse.hpp>
#include <random>
#include <sstream>
#include <sys/ioctl.h>
#include <lz4.h>
#include <libinput.h>
#include <libudev.h>
#include <linux/input-event-codes.h>
#include <linux/input.h>
#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>
#include <queue>
#include <mutex>

// Input state
static std::atomic<bool> running{true};
static std::atomic<int> current_mouse_x{0};
static std::atomic<int> current_mouse_y{0};
static std::atomic<int> screen_width{1920};
static std::atomic<int> screen_height{1080};

// Modifier tracking for exit combo and sending to server
static std::atomic<bool> shift_pressed{false};
static std::atomic<bool> ctrl_pressed{false};
static std::atomic<bool> alt_pressed{false};
static std::atomic<bool> delete_pressed{false};
static std::atomic<bool> x_pressed{false};
static std::atomic<bool> q_pressed{false};
static std::atomic<bool> i_pressed{false};
static std::atomic<bool> g_pressed{false};
static std::atomic<bool> plus_pressed{false};
static std::atomic<bool> minus_pressed{false};
static std::atomic<bool> zero_pressed{false};
static std::atomic<bool> f_pressed{false};
static std::atomic<bool> r_pressed{false};
static std::atomic<bool> c_pressed{false};
static std::atomic<bool> d_pressed{false};
static std::atomic<bool> s_pressed{false};
static std::atomic<bool> p_pressed{false};
static std::atomic<bool> a_pressed{false};
static std::atomic<bool> m_pressed{false};
static std::atomic<bool> equals_pressed{false};
static std::atomic<bool> pageup_pressed{false};
static std::atomic<bool> pagedown_pressed{false};
static std::atomic<bool> up_pressed{false};
static std::atomic<bool> down_pressed{false};
static std::atomic<bool> left_pressed{false};
static std::atomic<bool> right_pressed{false};
static std::atomic<bool> h_pressed{false};
static std::atomic<bool> t_pressed{false};
static std::atomic<bool> o_pressed{false};
static std::atomic<bool> w_pressed{false};
static std::atomic<bool> e_pressed{false};
static std::atomic<bool> n_pressed{false};
static std::atomic<bool> b_pressed{false};
static std::atomic<bool> v_pressed{false};
static std::atomic<bool> l_pressed{false};
static std::atomic<bool> home_pressed{false};
static std::atomic<bool> end_pressed{false};
static std::atomic<bool> leftbracket_pressed{false};
static std::atomic<bool> rightbracket_pressed{false};
static std::atomic<bool> backslash_pressed{false};
static std::atomic<bool> j_pressed{false};
static std::atomic<bool> k_pressed{false};
static std::atomic<bool> u_pressed{false};
static std::atomic<bool> y_pressed{false};
static std::atomic<bool> grave_pressed{false};
static std::atomic<bool> key1_pressed{false};
static std::atomic<bool> key2_pressed{false};
static std::atomic<bool> key3_pressed{false};
static std::atomic<bool> key4_pressed{false};

static std::atomic<bool> video_paused{false};
static std::atomic<bool> audio_muted{false};
static std::atomic<bool> microphone_muted{false};
static std::atomic<bool> input_forwarding_enabled{true};
static std::atomic<bool> exclusive_grab_enabled{false};

static std::atomic<bool> clear_screen_requested{false};
static std::atomic<int> skip_frames_counter{0};
static std::mutex clear_screen_mutex;

// Network
static int frame_socket = -1;
static int input_socket = -1;

// Feature flags
static bool feature_video = true;
static bool feature_audio = true;
static bool feature_input = true;
static bool feature_microphone = true;

// Last received frame for delta encoding
static std::string last_received_frame;

// libinput context
static struct libinput *li = nullptr;
static struct udev *udev = nullptr;
static bool exclusive_mode = false;

// Protocol definitions
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
    uint32_t format;
};

struct MicrophoneCapture
{
    pw_thread_loop *loop = nullptr;
    pw_stream *stream = nullptr;
    std::mutex mutex;
    std::queue<std::vector<uint8_t>> microphone_queue;
    std::atomic<bool> running{true};
    AudioFormat format{48000, 2, 1};
};

static MicrophoneCapture microphone_capture;
static int microphone_socket = -1;

struct AudioDataHeader
{
    uint32_t size;
    uint64_t timestamp_us;
};

struct AudioPlayback
{
    pw_thread_loop *loop = nullptr;
    pw_stream *stream = nullptr;
    std::mutex mutex;
    std::queue<std::vector<uint8_t>> audio_queue;
    std::atomic<bool> running{true};
    AudioFormat format{48000, 2, 1};
};

static AudioPlayback audio_playback;
static int audio_socket = -1;

static int config_socket = -1;

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
    char uuid[37];
};

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

struct ZoomState
{
    std::atomic<bool> enabled{false};
    std::atomic<bool> follow_mouse{true};
    std::atomic<double> zoom_level{2.0};
    std::atomic<int> view_width{800};
    std::atomic<int> view_height{600};
    std::atomic<int> center_x{960}; // Center of zoom viewport
    std::atomic<int> center_y{540};
    std::atomic<bool> smooth_pan{true};
    std::atomic<int> pan_speed{20}; // pixels per frame
};

struct ZoomConfig
{
    uint8_t enabled;
    uint8_t follow_mouse;
    double zoom_level;   // 1.0 - 10.0
    uint32_t view_width; // Viewport dimensions in screen pixels
    uint32_t view_height;
    int32_t center_x; // Center point of zoom
    int32_t center_y;
    uint8_t smooth_pan;
    uint32_t pan_speed; // Smoothing speed (pixels/frame)
};

static ZoomState zoom_state;

struct CompressedFrameHeader
{
    uint32_t compressed_size;
    uint32_t uncompressed_size;
};

struct RenderedFrameHeader
{
    uint32_t data_size;
};

struct ScreenInfo
{
    uint32_t width;
    uint32_t height;
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

static ClientConfig current_config;
static std::mutex config_mutex;

// Session ID generation
static std::string generate_uuid()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    std::uniform_int_distribution<> dis2(8, 11);

    std::ostringstream ss;
    ss << std::hex;
    for (int i = 0; i < 8; i++)
        ss << dis(gen);
    ss << "-";
    for (int i = 0; i < 4; i++)
        ss << dis(gen);
    ss << "-4";
    for (int i = 0; i < 3; i++)
        ss << dis(gen);
    ss << "-";
    ss << dis2(gen);
    for (int i = 0; i < 3; i++)
        ss << dis(gen);
    ss << "-";
    for (int i = 0; i < 12; i++)
        ss << dis(gen);

    return ss.str();
}

// Network helpers
static bool send_session_id(int socket, const std::string &session_id)
{
    MessageType type = MessageType::SESSION_ID;
    if (send(socket, &type, sizeof(type), MSG_NOSIGNAL) != sizeof(type))
        return false;

    SessionID sid;
    strncpy(sid.uuid, session_id.c_str(), sizeof(sid.uuid) - 1);
    sid.uuid[36] = '\0';

    if (send(socket, &sid, sizeof(sid), MSG_NOSIGNAL) != sizeof(sid))
        return false;

    return true;
}

static void send_zoom_config()
{
    if (config_socket < 0)
        return;

    MessageType type = MessageType::ZOOM_CONFIG;
    if (send(config_socket, &type, sizeof(type), MSG_NOSIGNAL) != sizeof(type))
    {
        return;
    }

    ZoomConfig config;
    config.enabled = zoom_state.enabled.load() ? 1 : 0;
    config.follow_mouse = zoom_state.follow_mouse.load() ? 1 : 0;
    config.zoom_level = zoom_state.zoom_level.load();
    config.view_width = zoom_state.view_width.load();
    config.view_height = zoom_state.view_height.load();
    config.center_x = zoom_state.center_x.load();
    config.center_y = zoom_state.center_y.load();
    config.smooth_pan = zoom_state.smooth_pan.load() ? 1 : 0;
    config.pan_speed = zoom_state.pan_speed.load();

    if (send(config_socket, &config, sizeof(config), MSG_NOSIGNAL) != sizeof(config))
    {
        std::cerr << "[ZOOM] Failed to send config\n";
        return;
    }

    std::cerr << "[ZOOM] Sent config: "
              << (config.enabled ? "ENABLED" : "DISABLED")
              << " level=" << config.zoom_level
              << " viewport=" << config.view_width << "x" << config.view_height
              << " follow=" << (config.follow_mouse ? "YES" : "NO") << "\n";
}

static bool send_client_config(const ClientConfig &config)
{
    if (config_socket < 0)
    {
        std::cerr << "[CONFIG] Socket not connected\n";
        return false;
    }

    MessageType type = MessageType::CLIENT_CONFIG;
    if (send(config_socket, &type, sizeof(type), MSG_NOSIGNAL) != sizeof(type))
    {
        std::cerr << "[CONFIG] Failed to send message type\n";
        return false;
    }
    if (send(config_socket, &config, sizeof(config), MSG_NOSIGNAL) != sizeof(config))
    {
        std::cerr << "[CONFIG] Failed to send config data\n";
        return false;
    }

    std::cerr << "[CONFIG] Sent: " << config.term_width << "x" << config.term_height
              << " fps=" << config.fps << " renderer=" << (int)config.renderer
              << " detail=" << (int)config.detail_level 
              << " follow_focus=" << (int)config.follow_focus << "\n";
    return true;
}

static void get_terminal_size(int &tw, int &th)
{
    struct winsize w{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0)
    {
        tw = w.ws_col;
        th = w.ws_row;
    }
    else
    {
        tw = 80;
        th = 24;
    }
}

static std::vector<uint8_t> decompress_frame(const std::vector<uint8_t> &compressed,
                                             size_t expected_size)
{
    std::vector<uint8_t> decompressed(expected_size);

    int result = LZ4_decompress_safe(
        (const char *)compressed.data(),
        (char *)decompressed.data(),
        compressed.size(),
        expected_size);

    if (result < 0)
    {
        std::cerr << "[DECOMPRESS] Failed: " << result << "\n";
        return {};
    }

    return decompressed;
}

static void on_process_playback(void *userdata)
{
    AudioPlayback *pb = static_cast<AudioPlayback *>(userdata);

    pw_buffer *b = pw_stream_dequeue_buffer(pb->stream);
    if (!b)
        return;

    spa_buffer *buf = b->buffer;
    if (!buf->datas[0].data)
    {
        pw_stream_queue_buffer(pb->stream, b);
        return;
    }

    uint8_t *dst = static_cast<uint8_t *>(buf->datas[0].data);
    uint32_t max_size = buf->datas[0].maxsize;
    uint32_t size = 0;

    std::lock_guard<std::mutex> lock(pb->mutex);

    // Check mute flag
    if (audio_muted.load() || pb->audio_queue.empty())
    {
        // Silence
        size = max_size;
        memset(dst, 0, size);
    }
    else
    {
        auto &audio_data = pb->audio_queue.front();
        size = std::min((uint32_t)audio_data.size(), max_size);
        memcpy(dst, audio_data.data(), size);
        pb->audio_queue.pop();
    }

    buf->datas[0].chunk->offset = 0;
    buf->datas[0].chunk->stride = pb->format.channels * sizeof(float);
    buf->datas[0].chunk->size = size;

    pw_stream_queue_buffer(pb->stream, b);
}

static const pw_stream_events playback_events = {
    .version = PW_VERSION_STREAM_EVENTS,
    .process = on_process_playback,
};

static bool init_audio_playback(const AudioFormat &fmt)
{
    if (!feature_audio)
        return true;

    pw_init(nullptr, nullptr);

    audio_playback.format = fmt;
    audio_playback.loop = pw_thread_loop_new("audio-playback", nullptr);
    if (!audio_playback.loop)
    {
        std::cerr << "[AUDIO] Failed to create thread loop\n";
        return false;
    }

    pw_thread_loop_lock(audio_playback.loop);

    pw_context *context = pw_context_new(
        pw_thread_loop_get_loop(audio_playback.loop), nullptr, 0);

    if (!context)
    {
        std::cerr << "[AUDIO] Failed to create context\n";
        pw_thread_loop_unlock(audio_playback.loop);
        return false;
    }

    audio_playback.stream = pw_stream_new_simple(
        pw_thread_loop_get_loop(audio_playback.loop),
        "wayterm-mirror-playback",
        pw_properties_new(
            PW_KEY_MEDIA_TYPE, "Audio",
            PW_KEY_MEDIA_CATEGORY, "Playback",
            PW_KEY_MEDIA_ROLE, "Music",
            nullptr),
        &playback_events,
        &audio_playback);

    if (!audio_playback.stream)
    {
        std::cerr << "[AUDIO] Failed to create stream\n";
        pw_thread_loop_unlock(audio_playback.loop);
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

    pw_stream_connect(audio_playback.stream,
                      PW_DIRECTION_OUTPUT,
                      PW_ID_ANY,
                      static_cast<pw_stream_flags>(
                          PW_STREAM_FLAG_AUTOCONNECT |
                          PW_STREAM_FLAG_MAP_BUFFERS |
                          PW_STREAM_FLAG_RT_PROCESS),
                      params, 1);

    pw_thread_loop_unlock(audio_playback.loop);
    pw_thread_loop_start(audio_playback.loop);

    std::cerr << "[AUDIO] Playback initialized (" << fmt.sample_rate
              << "Hz " << (int)fmt.channels << "ch)\n";
    return true;
}

static void cleanup_audio_playback()
{
    if (!feature_audio)
        return;

    audio_playback.running = false;

    if (audio_playback.stream)
    {
        pw_stream_destroy(audio_playback.stream);
    }

    if (audio_playback.loop)
    {
        pw_thread_loop_stop(audio_playback.loop);
        pw_thread_loop_destroy(audio_playback.loop);
    }

    pw_deinit();
    std::cerr << "[AUDIO] Playback cleaned up\n";
}

static void on_process_microphone_capture(void *userdata)
{
    MicrophoneCapture *cap = static_cast<MicrophoneCapture *>(userdata);

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
        std::vector<uint8_t> microphone_data(src, src + size);

        std::lock_guard<std::mutex> lock(cap->mutex);
        cap->microphone_queue.push(std::move(microphone_data));

        while (cap->microphone_queue.size() > 10)
        {
            cap->microphone_queue.pop();
        }
    }

    pw_stream_queue_buffer(cap->stream, b);
}

static const pw_stream_events microphone_capture_events = {
    .version = PW_VERSION_STREAM_EVENTS,
    .process = on_process_microphone_capture,
};

static bool init_microphone_capture()
{
    if (!feature_microphone)
        return true;

    microphone_capture.loop = pw_thread_loop_new("mic-capture", nullptr);
    if (!microphone_capture.loop)
    {
        std::cerr << "[MICROPHONE OUT] Failed to create thread loop\n";
        return false;
    }

    pw_thread_loop_lock(microphone_capture.loop);

    pw_context *context = pw_context_new(
        pw_thread_loop_get_loop(microphone_capture.loop), nullptr, 0);

    if (!context)
    {
        std::cerr << "[MICROPHONE OUT] Failed to create context\n";
        pw_thread_loop_unlock(microphone_capture.loop);
        return false;
    }

    microphone_capture.stream = pw_stream_new_simple(
        pw_thread_loop_get_loop(microphone_capture.loop),
        "wayterm-mirror-mic-capture",
        pw_properties_new(
            PW_KEY_MEDIA_TYPE, "Audio",
            PW_KEY_MEDIA_CATEGORY, "Capture",
            PW_KEY_MEDIA_ROLE, "Communication",
            nullptr),
        &microphone_capture_events,
        &microphone_capture);

    if (!microphone_capture.stream)
    {
        std::cerr << "[MICROPHONE OUT] Failed to create stream\n";
        pw_thread_loop_unlock(microphone_capture.loop);
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

    pw_stream_connect(microphone_capture.stream,
                      PW_DIRECTION_INPUT,
                      PW_ID_ANY,
                      static_cast<pw_stream_flags>(
                          PW_STREAM_FLAG_AUTOCONNECT |
                          PW_STREAM_FLAG_MAP_BUFFERS |
                          PW_STREAM_FLAG_RT_PROCESS),
                      params, 1);

    pw_thread_loop_unlock(microphone_capture.loop);
    pw_thread_loop_start(microphone_capture.loop);

    std::cerr << "[MICROPHONE OUT] Microphone capture initialized (48kHz stereo F32LE)\n";
    return true;
}

static void microphone_send_thread()
{
    if (!feature_microphone || microphone_socket < 0)
        return;

    std::cerr << "[MICROPHONE OUT] Send thread started\n";

    if (!init_microphone_capture())
    {
        std::cerr << "[MICROPHONE OUT] Failed to init capture\n";
        return;
    }

    // Send format
    {
        MessageType type = MessageType::MICROPHONE_FORMAT;
        send(microphone_socket, &type, sizeof(type), MSG_NOSIGNAL);
        send(microphone_socket, &microphone_capture.format, sizeof(microphone_capture.format), MSG_NOSIGNAL);
    }

    while (running && microphone_capture.running)
    {
        std::vector<uint8_t> microphone_data;

        {
            std::lock_guard<std::mutex> lock(microphone_capture.mutex);
            if (!microphone_capture.microphone_queue.empty())
            {
                microphone_data = std::move(microphone_capture.microphone_queue.front());
                microphone_capture.microphone_queue.pop();
            }
        }

        if (!microphone_data.empty())
        {
            // Check mute flag - send silence if muted
            if (microphone_muted.load())
            {
                memset(microphone_data.data(), 0, microphone_data.size());
            }

            MessageType type = MessageType::MICROPHONE_DATA;
            AudioDataHeader header;
            header.size = microphone_data.size();
            header.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now().time_since_epoch())
                                      .count();

            if (send(microphone_socket, &type, sizeof(type), MSG_NOSIGNAL) != sizeof(type) ||
                send(microphone_socket, &header, sizeof(header), MSG_NOSIGNAL) != sizeof(header) ||
                send(microphone_socket, microphone_data.data(), microphone_data.size(), MSG_NOSIGNAL) != (ssize_t)microphone_data.size())
            {
                std::cerr << "[MICROPHONE OUT] Send failed\n";
                break;
            }
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    std::cerr << "[MICROPHONE OUT] Send thread stopped\n";
}

static void cleanup_microphone_capture()
{
    if (!feature_microphone)
        return;

    microphone_capture.running = false;

    if (microphone_capture.stream)
    {
        pw_stream_destroy(microphone_capture.stream);
    }

    if (microphone_capture.loop)
    {
        pw_thread_loop_stop(microphone_capture.loop);
        pw_thread_loop_destroy(microphone_capture.loop);
    }

    std::cerr << "[MICROPHONE OUT] Capture cleaned up\n";
}

static void audio_receive_thread()
{
    if (!feature_audio || audio_socket < 0)
        return;

    std::cerr << "[AUDIO] Receive thread started\n";

    // Receive format
    MessageType type;
    AudioFormat fmt;

    if (recv(audio_socket, &type, sizeof(type), 0) != sizeof(type) ||
        type != MessageType::AUDIO_FORMAT ||
        recv(audio_socket, &fmt, sizeof(fmt), 0) != sizeof(fmt))
    {
        std::cerr << "[AUDIO] Failed to receive format\n";
        return;
    }

    if (!init_audio_playback(fmt))
    {
        std::cerr << "[AUDIO] Failed to init playback\n";
        return;
    }

    while (running && audio_playback.running)
    {
        if (recv(audio_socket, &type, sizeof(type), 0) != sizeof(type))
            break;

        if (type != MessageType::AUDIO_DATA)
        {
            std::cerr << "[AUDIO] Unexpected message type\n";
            continue;
        }

        AudioDataHeader header;
        if (recv(audio_socket, &header, sizeof(header), 0) != sizeof(header))
            break;

        std::vector<uint8_t> audio_data(header.size);
        size_t total = 0;
        while (total < header.size)
        {
            ssize_t n = recv(audio_socket, audio_data.data() + total,
                             header.size - total, 0);
            if (n <= 0)
                goto cleanup;
            total += n;
        }

        {
            std::lock_guard<std::mutex> lock(audio_playback.mutex);
            audio_playback.audio_queue.push(std::move(audio_data));

            // Prevent queue buildup
            while (audio_playback.audio_queue.size() > 10)
            {
                audio_playback.audio_queue.pop();
            }
        }
    }

cleanup:
    std::cerr << "[AUDIO] Receive thread stopped\n";
}

static bool receive_newest_frame(std::string &rendered)
{
    struct pollfd pfd = {frame_socket, POLLIN, 0};

    int ret = poll(&pfd, 1, 0);  // Non-blocking check
    if (ret <= 0)
        return false;

    std::vector<uint8_t> latest_data;
    bool got_frame = false;

    while (true)
    {
        struct pollfd check_pfd = {frame_socket, POLLIN, 0};
        if (poll(&check_pfd, 1, 0) <= 0)
            break;

        MessageType type;
        ssize_t n = recv(frame_socket, &type, sizeof(type), 0);
        if (n != sizeof(type))
        {
            if (got_frame)
                break;
            return false;
        }

        if (type == MessageType::SCREEN_INFO)
        {
            ScreenInfo info;
            n = recv(frame_socket, &info, sizeof(info), 0);
            if (n == sizeof(info))
            {
                screen_width = info.width;
                screen_height = info.height;
            }
            continue;
        }

        // === NEW: Handle DELTA_FRAME ===
        if (type == MessageType::DELTA_FRAME)
        {
            // Receive delta header
            DeltaFrameHeader header;
            size_t total = 0;
            while (total < sizeof(header))
            {
                n = recv(frame_socket, ((uint8_t *)&header) + total,
                         sizeof(header) - total, 0);
                if (n <= 0)
                    return false;
                total += n;
            }

            if (last_received_frame.size() != header.base_frame_size)
            {
                std::cerr << "[DELTA] Size mismatch, skipping delta\n";
                // Skip this delta - read and discard the changes
                for (uint32_t i = 0; i < header.num_changes; i++)
                {
                    FrameChange change;
                    recv(frame_socket, &change, sizeof(change), 0);
                    std::vector<uint8_t> discard(change.length);
                    recv(frame_socket, discard.data(), change.length, 0);
                }
                continue;
            }

            // Apply delta to last frame
            std::string result = last_received_frame;

            for (uint32_t i = 0; i < header.num_changes; i++)
            {
                FrameChange change;
                total = 0;
                while (total < sizeof(change))
                {
                    n = recv(frame_socket, ((uint8_t *)&change) + total,
                             sizeof(change) - total, 0);
                    if (n <= 0)
                        return false;
                    total += n;
                }

                // Receive changed data
                std::vector<uint8_t> data(change.length);
                total = 0;
                while (total < change.length)
                {
                    n = recv(frame_socket, data.data() + total,
                             change.length - total, 0);
                    if (n <= 0)
                        return false;
                    total += n;
                }

                // Apply change
                if (change.offset + change.length <= result.size())
                {
                    memcpy(&result[change.offset], data.data(), change.length);
                }
                else
                {
                    std::cerr << "[DELTA] Invalid offset/length\n";
                }
            }

            latest_data.assign(result.begin(), result.end());
            got_frame = true;
            continue;
        }

        if (type == MessageType::COMPRESSED_FRAME)
        {
            CompressedFrameHeader comp_header;
            size_t total = 0;
            while (total < sizeof(comp_header))
            {
                n = recv(frame_socket, ((uint8_t *)&comp_header) + total,
                         sizeof(comp_header) - total, 0);
                if (n <= 0)
                    return false;
                total += n;
            }

            std::vector<uint8_t> compressed(comp_header.compressed_size);
            total = 0;
            while (total < comp_header.compressed_size)
            {
                n = recv(frame_socket, compressed.data() + total,
                         comp_header.compressed_size - total, 0);
                if (n <= 0)
                    return false;
                total += n;
            }

            std::vector<uint8_t> decompressed = decompress_frame(compressed,
                                                                 comp_header.uncompressed_size);
            if (decompressed.empty())
                continue;

            size_t offset = 0;
            MessageType inner_type = static_cast<MessageType>(decompressed[offset]);
            offset += sizeof(MessageType);

            // === NEW: Handle compressed delta frames ===
            if (inner_type == MessageType::DELTA_FRAME)
            {
                DeltaFrameHeader header;
                memcpy(&header, decompressed.data() + offset, sizeof(header));
                offset += sizeof(header);

                if (last_received_frame.size() != header.base_frame_size)
                {
                    std::cerr << "[DELTA] Compressed: Size mismatch\n";
                    continue;
                }

                std::string result = last_received_frame;

                for (uint32_t i = 0; i < header.num_changes; i++)
                {
                    FrameChange change;
                    memcpy(&change, decompressed.data() + offset, sizeof(change));
                    offset += sizeof(change);

                    if (change.offset + change.length <= result.size())
                    {
                        memcpy(&result[change.offset], decompressed.data() + offset, change.length);
                    }
                    offset += change.length;
                }

                latest_data.assign(result.begin(), result.end());
                got_frame = true;
            }
            else if (inner_type == MessageType::RENDERED_FRAME)
            {
                RenderedFrameHeader frame_header;
                memcpy(&frame_header, decompressed.data() + offset, sizeof(frame_header));
                offset += sizeof(frame_header);

                latest_data.assign(decompressed.begin() + offset,
                                   decompressed.begin() + offset + frame_header.data_size);
                got_frame = true;
            }

            continue;
        }

        if (type == MessageType::RENDERED_FRAME)
        {
            RenderedFrameHeader header;
            size_t total = 0;
            while (total < sizeof(header))
            {
                n = recv(frame_socket, ((uint8_t *)&header) + total,
                         sizeof(header) - total, 0);
                if (n <= 0)
                    return false;
                total += n;
            }

            std::vector<uint8_t> data(header.data_size);
            total = 0;
            while (total < header.data_size)
            {
                n = recv(frame_socket, data.data() + total,
                         header.data_size - total, 0);
                if (n <= 0)
                    return false;
                total += n;
            }

            latest_data = std::move(data);
            got_frame = true;
        }
    }

    if (got_frame)
    {
        rendered = std::string(latest_data.begin(), latest_data.end());
        last_received_frame = rendered; // Cache for delta decoding
        return true;
    }

    return false;
}

// libinput helpers
static int open_restricted(const char *path, int flags, void *user_data)
{
    int fd = open(path, flags);

    if (fd < 0)
    {
        return -errno;
    }

    std::cerr << "[INPUT] Opened device: " << path << "\n";

    // If exclusive mode is enabled, grab the device using EVIOCGRAB
    if (exclusive_mode)
    {
        std::cerr << "[INPUT] Attempting to grab device exclusively: " << path << "\n";

        // EVIOCGRAB prevents other clients (including X11/Wayland) from receiving events
        int grab = 1;
        if (ioctl(fd, EVIOCGRAB, (void *)&grab) < 0)
        {
            std::cerr << "[INPUT] WARNING: Failed to grab device: " << path
                      << " (" << strerror(errno) << ")\n";
            std::cerr << "[INPUT] Device will work but may still send events to host OS\n";
        }
        else
        {
            std::cerr << "[INPUT] Successfully grabbed exclusive access: " << path << "\n";
        }
    }

    return fd;
}

static void close_restricted(int fd, void *user_data)
{
    close(fd);
}

static const struct libinput_interface interface = {
    .open_restricted = open_restricted,
    .close_restricted = close_restricted,
};

// Input event handlers

static void process_libinput_events();

static void toggle_exclusive_grab()
{
    exclusive_grab_enabled = !exclusive_grab_enabled.load();
    std::cerr << "[INPUT] Exclusive grab: " << (exclusive_grab_enabled.load() ? "ON" : "OFF") << "\n";

    if (!li)
        return;

    exclusive_mode = exclusive_grab_enabled.load();
    libinput_dispatch(li);
    process_libinput_events();
}

static void cycle_renderer()
{
    std::lock_guard<std::mutex> lock(config_mutex);
    current_config.renderer = (current_config.renderer + 1) % 4;

    const char *names[] = {"braille", "blocks", "ascii", "hybrid"};
    std::cerr << "[RENDERER] Switched to: " << names[current_config.renderer] << "\n";

    get_terminal_size((int &)current_config.term_width, (int &)current_config.term_height);
    send_client_config(current_config);
}

static void cycle_color_mode()
{
    std::lock_guard<std::mutex> lock(config_mutex);
    current_config.color_mode = (current_config.color_mode + 1) % 3;

    const char *names[] = {"16-color", "256-color", "truecolor"};
    std::cerr << "[COLOR] Switched to: " << names[current_config.color_mode] << "\n";

    get_terminal_size((int &)current_config.term_width, (int &)current_config.term_height);
    send_client_config(current_config);
}

static void adjust_detail(int delta)
{
    std::lock_guard<std::mutex> lock(config_mutex);
    current_config.detail_level = std::clamp((int)current_config.detail_level + delta, 0, 100);

    std::cerr << "[DETAIL] Level: " << (int)current_config.detail_level << "\n";

    get_terminal_size((int &)current_config.term_width, (int &)current_config.term_height);
    send_client_config(current_config);
}

static void adjust_zoom(double delta)
{
    double new_level = std::clamp(zoom_state.zoom_level.load() + delta, 1.0, 10.0);
    zoom_state.zoom_level = new_level;
    std::cerr << "[ZOOM] Level: " << new_level << "x\n";
    send_zoom_config();
}

static void adjust_quality(int delta)
{
    std::lock_guard<std::mutex> lock(config_mutex);
    current_config.quality = std::clamp((int)current_config.quality + delta, 0, 100);
    std::cerr << "[QUALITY] Level: " << (int)current_config.quality << "\n";
    get_terminal_size((int &)current_config.term_width, (int &)current_config.term_height);
    send_client_config(current_config);
}

static void adjust_rotation(double delta)
{
    std::lock_guard<std::mutex> lock(config_mutex);
    current_config.rotation_angle = fmod(current_config.rotation_angle + delta + 360.0, 360.0);
    std::cerr << "[ROTATION] Angle: " << current_config.rotation_angle << "°\n";
    get_terminal_size((int &)current_config.term_width, (int &)current_config.term_height);
    send_client_config(current_config);
}

static void set_rotation(double angle)
{
    std::lock_guard<std::mutex> lock(config_mutex);
    current_config.rotation_angle = fmod(angle + 360.0, 360.0);
    std::cerr << "[ROTATION] Angle: " << current_config.rotation_angle << "°\n";
    get_terminal_size((int &)current_config.term_width, (int &)current_config.term_height);
    send_client_config(current_config);
}

static void adjust_fps(int delta)
{
    std::lock_guard<std::mutex> lock(config_mutex);
    current_config.fps = std::clamp((int)current_config.fps + delta, 0, 120);
    std::cerr << "[FPS] Target: " << current_config.fps << "\n";
    get_terminal_size((int &)current_config.term_width, (int &)current_config.term_height);
    send_client_config(current_config);
}

static void cycle_output()
{
    std::lock_guard<std::mutex> lock(config_mutex);
    // Toggle between follow mode (0xFFFFFFFF) and output 0, 1, 2...
    if (current_config.follow_focus)
    {
        current_config.follow_focus = 0;
        current_config.output_index = 0;
        std::cerr << "[OUTPUT] Switched to output 0 (manual)\n";
    }
    else
    {
        current_config.output_index = (current_config.output_index + 1) % 4;
        if (current_config.output_index == 0)
        {
            current_config.follow_focus = 1;
            std::cerr << "[OUTPUT] Switched to follow-focus mode\n";
        }
        else
        {
            std::cerr << "[OUTPUT] Switched to output " << current_config.output_index << "\n";
        }
    }
    get_terminal_size((int &)current_config.term_width, (int &)current_config.term_height);
    send_client_config(current_config);
}

static void set_renderer(uint8_t renderer)
{
    std::lock_guard<std::mutex> lock(config_mutex);
    current_config.renderer = renderer % 4;
    const char *names[] = {"braille", "blocks", "ascii", "hybrid"};
    std::cerr << "[RENDERER] Switched to: " << names[current_config.renderer] << "\n";
    get_terminal_size((int &)current_config.term_width, (int &)current_config.term_height);
    send_client_config(current_config);
}

static void print_shortcuts_help()
{
    std::lock_guard<std::mutex> lock(config_mutex);
    std::cerr << "\n";
    std::cerr << "╔════════════════════════════════════════════════════════════════════════╗\n";
    std::cerr << "║               WAYTERMIRROR CLIENT KEYBOARD SHORTCUTS                   ║\n";
    std::cerr << "╠════════════════════════════════════════════════════════════════════════╣\n";
    std::cerr << "║  All shortcuts use Ctrl+Alt+Shift as modifier prefix                   ║\n";
    std::cerr << "╠════════════════════════════════════════════════════════════════════════╣\n";
    std::cerr << "║ SESSION CONTROL                                                        ║\n";
    std::cerr << "║   Q            Quit / disconnect gracefully                            ║\n";
    std::cerr << "║   H            Show this help                                          ║\n";
    std::cerr << "║   P            Pause / resume video rendering                          ║\n";
    std::cerr << "╠════════════════════════════════════════════════════════════════════════╣\n";
    std::cerr << "║ INPUT CONTROL                                                          ║\n";
    std::cerr << "║   I            Toggle input forwarding to server                       ║\n";
    std::cerr << "║   G            Toggle exclusive grab (EVIOCGRAB)                       ║\n";
    std::cerr << "╠════════════════════════════════════════════════════════════════════════╣\n";
    std::cerr << "║ ZOOM CONTROL                                                           ║\n";
    std::cerr << "║   Z            Toggle zoom mode                                        ║\n";
    std::cerr << "║   + / =        Zoom in (+0.5x)                                         ║\n";
    std::cerr << "║   -            Zoom out (-0.5x)                                        ║\n";
    std::cerr << "║   0            Reset zoom to default (2.0x)                            ║\n";
    std::cerr << "║   N            Toggle zoom follow mouse                                ║\n";
    std::cerr << "║   Arrow Keys   Pan viewport (20px per press)                           ║\n";
    std::cerr << "║   PageUp/Dn    Fast vertical pan (100px per press)                     ║\n";
    std::cerr << "║   Home/End     Fast horizontal pan (100px per press)                   ║\n";
    std::cerr << "╠════════════════════════════════════════════════════════════════════════╣\n";
    std::cerr << "║ RENDERING                                                              ║\n";
    std::cerr << "║   R            Cycle renderer (braille→blocks→ascii→hybrid)            ║\n";
    std::cerr << "║   1/2/3/4      Quick switch: braille/blocks/ascii/hybrid               ║\n";
    std::cerr << "║   C            Cycle color mode (16→256→truecolor)                     ║\n";
    std::cerr << "║   D / S        Increase / Decrease detail level (±10)                  ║\n";
    std::cerr << "║   W / E        Increase / Decrease quality (±10)                       ║\n";
    std::cerr << "║   O            Toggle smooth panning                                   ║\n";
    std::cerr << "║   B            Toggle keep aspect ratio                                ║\n";
    std::cerr << "║   V            Cycle render device (CPU→CUDA)                          ║\n";
    std::cerr << "║   L            Cycle compression (off→LZ4→LZ4 HC)                      ║\n";
    std::cerr << "╠════════════════════════════════════════════════════════════════════════╣\n";
    std::cerr << "║ ROTATION                                                               ║\n";
    std::cerr << "║   [            Rotate left (-15°)                                      ║\n";
    std::cerr << "║   ]            Rotate right (+15°)                                     ║\n";
    std::cerr << "║   \\            Reset rotation to 0°                                    ║\n";
    std::cerr << "║   T            Rotate 90° clockwise                                    ║\n";
    std::cerr << "║   Y            Rotate 90° counter-clockwise                            ║\n";
    std::cerr << "╠════════════════════════════════════════════════════════════════════════╣\n";
    std::cerr << "║ FPS / OUTPUT                                                           ║\n";
    std::cerr << "║   J / K        Increase / Decrease target FPS (±5)                     ║\n";
    std::cerr << "║   `            Cycle output (0→1→2→follow→...)                         ║\n";
    std::cerr << "║   U            Toggle compression on/off                               ║\n";
    std::cerr << "╠════════════════════════════════════════════════════════════════════════╣\n";
    std::cerr << "║ AUDIO/VIDEO                                                            ║\n";
    std::cerr << "║   A            Toggle audio playback (mute/unmute)                     ║\n";
    std::cerr << "║   M            Toggle microphone capture (mute/unmute)                 ║\n";
    std::cerr << "║   F            Toggle follow-focus mode                                ║\n";
    std::cerr << "╠════════════════════════════════════════════════════════════════════════╣\n";
    std::cerr << "║ CURRENT STATE                                                          ║\n";
    std::cerr << "║   Renderer:       " << std::setw(8) << std::left << (const char*[]){"braille", "blocks", "ascii", "hybrid"}[current_config.renderer] << "  Color: " << std::setw(9) << (const char*[]){"16", "256", "truecolor"}[current_config.color_mode] << "  Device: " << std::setw(4) << (current_config.render_device ? "CUDA" : "CPU") << "   ║\n";
    std::cerr << "║   Detail: " << std::setw(3) << (int)current_config.detail_level << "       Quality: " << std::setw(3) << (int)current_config.quality << "       FPS: " << std::setw(3) << current_config.fps << "             ║\n";
    std::cerr << "║   Rotation: " << std::setw(5) << std::fixed << std::setprecision(0) << current_config.rotation_angle << "°   Aspect: " << (current_config.keep_aspect_ratio ? "ON " : "OFF") << "        Compress: " << (current_config.compress ? "ON " : "OFF") << "           ║\n";
    std::cerr << "║   Input Fwd: " << (input_forwarding_enabled.load() ? "ON " : "OFF") << "      Exclusive: " << (exclusive_grab_enabled.load() ? "ON " : "OFF") << "       Zoom: " << (zoom_state.enabled.load() ? "ON " : "OFF") << " (" << std::setprecision(1) << zoom_state.zoom_level.load() << "x)     ║\n";
    std::cerr << "║   Video: " << (video_paused.load() ? "PAUSED " : "RUNNING") << "    Audio: " << (audio_muted.load() ? "MUTED  " : "PLAYING") << "       Mic: " << (microphone_muted.load() ? "MUTED  " : "CAPTURE") << "        ║\n";
    std::cerr << "╚════════════════════════════════════════════════════════════════════════╝\n";
    std::cerr << "\n";
}

static void pan_zoom(int dx, int dy)
{
    if (!zoom_state.enabled.load())
        return;

    int new_x = std::clamp(zoom_state.center_x.load() + dx, 0, screen_width.load() - 1);
    int new_y = std::clamp(zoom_state.center_y.load() + dy, 0, screen_height.load() - 1);

    zoom_state.center_x = new_x;
    zoom_state.center_y = new_y;
    send_zoom_config();
}

static void send_key_event(uint32_t keycode, bool pressed)
{
    bool is_shift = (keycode == KEY_LEFTSHIFT || keycode == KEY_RIGHTSHIFT);
    bool is_ctrl = (keycode == KEY_LEFTCTRL || keycode == KEY_RIGHTCTRL);
    bool is_alt = (keycode == KEY_LEFTALT || keycode == KEY_RIGHTALT);
    bool is_delete = (keycode == KEY_DELETE);
    bool is_x = (keycode == KEY_X);
    bool is_z = (keycode == KEY_Z);
    bool is_q = (keycode == KEY_Q);
    bool is_i = (keycode == KEY_I);
    bool is_g = (keycode == KEY_G);
    bool is_plus = (keycode == KEY_KPPLUS);
    bool is_minus = (keycode == KEY_KPMINUS || keycode == KEY_MINUS);
    bool is_zero = (keycode == KEY_0);
    bool is_equals = (keycode == KEY_EQUAL);
    bool is_f = (keycode == KEY_F);
    bool is_r = (keycode == KEY_R);
    bool is_c = (keycode == KEY_C);
    bool is_d = (keycode == KEY_D);
    bool is_s = (keycode == KEY_S);
    bool is_p = (keycode == KEY_P);
    bool is_a = (keycode == KEY_A);
    bool is_m = (keycode == KEY_M);
    bool is_pageup = (keycode == KEY_PAGEUP);
    bool is_pagedown = (keycode == KEY_PAGEDOWN);
    bool is_up = (keycode == KEY_UP);
    bool is_down = (keycode == KEY_DOWN);
    bool is_left = (keycode == KEY_LEFT);
    bool is_right = (keycode == KEY_RIGHT);
    bool is_h = (keycode == KEY_H);
    bool is_t = (keycode == KEY_T);
    bool is_o = (keycode == KEY_O);
    bool is_w = (keycode == KEY_W);
    bool is_e = (keycode == KEY_E);
    bool is_n = (keycode == KEY_N);
    bool is_b = (keycode == KEY_B);
    bool is_v = (keycode == KEY_V);
    bool is_l = (keycode == KEY_L);
    bool is_home = (keycode == KEY_HOME);
    bool is_end = (keycode == KEY_END);
    bool is_leftbracket = (keycode == KEY_LEFTBRACE);
    bool is_rightbracket = (keycode == KEY_RIGHTBRACE);
    bool is_backslash = (keycode == KEY_BACKSLASH);
    bool is_j = (keycode == KEY_J);
    bool is_k = (keycode == KEY_K);
    bool is_u = (keycode == KEY_U);
    bool is_y = (keycode == KEY_Y);
    bool is_grave = (keycode == KEY_GRAVE);
    bool is_1 = (keycode == KEY_1);
    bool is_2 = (keycode == KEY_2);
    bool is_3 = (keycode == KEY_3);
    bool is_4 = (keycode == KEY_4);

    // Track modifier state
    if (is_shift)
        shift_pressed = pressed;
    if (is_ctrl)
        ctrl_pressed = pressed;
    if (is_alt)
        alt_pressed = pressed;
    if (is_delete)
        delete_pressed = pressed;
    if (is_x)
        x_pressed = pressed;

    bool combo = shift_pressed.load() && ctrl_pressed.load() && alt_pressed.load();

    // Check for all shortcuts on key press
    if (pressed && combo)
    {
        // === SESSION CONTROL ===

        // Quit (Ctrl+Alt+Shift+Q)
        if (is_q)
        {
            std::cerr << "\n[EXIT] Quit shortcut detected!\n";
            running = false;
            return;
        }

        // Show help (Ctrl+Alt+Shift+H)
        if (is_h)
        {
            print_shortcuts_help();
            return;
        }

        // Pause/resume video (Ctrl+Alt+Shift+P)
        if (is_p)
        {
            video_paused = !video_paused.load();
            std::cerr << "[VIDEO] " << (video_paused.load() ? "PAUSED" : "RESUMED") << "\n";
            return;
        }

        // === INPUT CONTROL ===

        // Toggle input forwarding (Ctrl+Alt+Shift+I)
        if (is_i)
        {
            input_forwarding_enabled = !input_forwarding_enabled.load();
            std::cerr << "[INPUT] Forwarding: " << (input_forwarding_enabled.load() ? "ON" : "OFF") << "\n";
            return;
        }

        // Toggle exclusive grab (Ctrl+Alt+Shift+G)
        if (is_g)
        {
            toggle_exclusive_grab();
            return;
        }

        // === ZOOM CONTROL ===

        // Toggle zoom (Ctrl+Alt+Shift+Z)
        if (is_z)
        {
            {
                std::lock_guard<std::mutex> lock2(clear_screen_mutex);
                clear_screen_requested.store(true);
                skip_frames_counter.store(5);  // Skip 5 frames to allow server to process config
            }
            zoom_state.enabled = !zoom_state.enabled.load();
            std::cerr << "[ZOOM] " << (zoom_state.enabled.load() ? "ENABLED" : "DISABLED") << " (" << zoom_state.zoom_level.load() << "x)\n";
            send_zoom_config();
            return;
        }

        // Zoom in (Ctrl+Alt+Shift++ or =)
        if (is_plus || is_equals)
        {
            adjust_zoom(0.5);
            return;
        }

        // Zoom out (Ctrl+Alt+Shift+-)
        if (is_minus)
        {
            adjust_zoom(-0.5);
            return;
        }

        // Reset zoom (Ctrl+Alt+Shift+0)
        if (is_zero)
        {
            zoom_state.zoom_level = 2.0;
            zoom_state.center_x = screen_width.load() / 2;
            zoom_state.center_y = screen_height.load() / 2;
            std::cerr << "[ZOOM] Reset to 2.0x (centered)\n";
            send_zoom_config();
            return;
        }

        // Toggle zoom follow mouse (Ctrl+Alt+Shift+N)
        if (is_n)
        {
            zoom_state.follow_mouse = !zoom_state.follow_mouse.load();
            std::cerr << "[ZOOM] Follow mouse: " << (zoom_state.follow_mouse.load() ? "ON" : "OFF") << "\n";
            send_zoom_config();
            return;
        }

        // Pan zoom with arrow keys (normal speed)
        if (is_left)
        {
            pan_zoom(-zoom_state.pan_speed.load(), 0);
            return;
        }
        if (is_right)
        {
            pan_zoom(zoom_state.pan_speed.load(), 0);
            return;
        }
        if (is_up)
        {
            pan_zoom(0, -zoom_state.pan_speed.load());
            return;
        }
        if (is_down)
        {
            pan_zoom(0, zoom_state.pan_speed.load());
            return;
        }

        // Fast vertical pan with PageUp/PageDown
        if (is_pageup)
        {
            pan_zoom(0, -zoom_state.pan_speed.load() * 5);
            return;
        }
        if (is_pagedown)
        {
            pan_zoom(0, zoom_state.pan_speed.load() * 5);
            return;
        }

        // Fast horizontal pan with Home/End
        if (is_home)
        {
            pan_zoom(-zoom_state.pan_speed.load() * 5, 0);
            return;
        }
        if (is_end)
        {
            pan_zoom(zoom_state.pan_speed.load() * 5, 0);
            return;
        }

        // === RENDERING ===

        // Cycle renderer (Ctrl+Alt+Shift+R)
        if (is_r)
        {
            cycle_renderer();
            return;
        }

        // Cycle color mode (Ctrl+Alt+Shift+C)
        if (is_c)
        {
            cycle_color_mode();
            return;
        }

        // Increase detail (Ctrl+Alt+Shift+D)
        if (is_d)
        {
            adjust_detail(10);
            return;
        }

        // Decrease detail (Ctrl+Alt+Shift+S)
        if (is_s)
        {
            adjust_detail(-10);
            return;
        }

        // Increase quality (Ctrl+Alt+Shift+W)
        if (is_w)
        {
            adjust_quality(10);
            return;
        }

        // Decrease quality (Ctrl+Alt+Shift+E)
        if (is_e)
        {
            adjust_quality(-10);
            return;
        }

        // Toggle smooth panning (Ctrl+Alt+Shift+O)
        if (is_o)
        {
            zoom_state.smooth_pan = !zoom_state.smooth_pan.load();
            std::cerr << "[ZOOM] Smooth panning: " << (zoom_state.smooth_pan.load() ? "ON" : "OFF") << "\n";
            send_zoom_config();
            return;
        }

        // Toggle keep aspect ratio (Ctrl+Alt+Shift+B)
        if (is_b)
        {
            std::lock_guard<std::mutex> lock(config_mutex);
            current_config.keep_aspect_ratio = !current_config.keep_aspect_ratio;
            std::cerr << "[DISPLAY] Keep aspect ratio: " << (current_config.keep_aspect_ratio ? "ON" : "OFF") << "\n";
            {
                std::lock_guard<std::mutex> lock2(clear_screen_mutex);
                clear_screen_requested.store(true);
                skip_frames_counter.store(5);  // Skip 5 frames to allow server to process config
            }
            get_terminal_size((int &)current_config.term_width, (int &)current_config.term_height);
            send_client_config(current_config);
            return;
        }

        // Cycle render device (Ctrl+Alt+Shift+V)
        if (is_v)
        {
            std::lock_guard<std::mutex> lock(config_mutex);
            current_config.render_device = (current_config.render_device + 1) % 2;
            const char *devices[] = {"CPU", "CUDA"};
            std::cerr << "[RENDER] Device: " << devices[current_config.render_device] << "\n";
            {
                std::lock_guard<std::mutex> lock2(clear_screen_mutex);
                clear_screen_requested.store(true);
                skip_frames_counter.store(5);  // Skip 5 frames to allow server to process config
            }
            get_terminal_size((int &)current_config.term_width, (int &)current_config.term_height);
            send_client_config(current_config);
            return;
        }

        // === AUDIO/VIDEO ===

        // Toggle focus-follow (Ctrl+Alt+Shift+F)
        if (is_f)
        {
            std::lock_guard<std::mutex> lock(config_mutex);
            current_config.follow_focus = !current_config.follow_focus;
            std::cerr << "[FOCUS] Focus-follow: " << (current_config.follow_focus ? "ON" : "OFF") << "\n";
            get_terminal_size((int &)current_config.term_width, (int &)current_config.term_height);
            send_client_config(current_config);
            return;
        }

        // Toggle audio (Ctrl+Alt+Shift+A)
        if (is_a)
        {
            audio_muted = !audio_muted.load();
            std::cerr << "[AUDIO] " << (audio_muted.load() ? "MUTED" : "UNMUTED") << "\n";
            return;
        }

        // Toggle microphone (Ctrl+Alt+Shift+M)
        if (is_m)
        {
            microphone_muted = !microphone_muted.load();
            std::cerr << "[MICROPHONE] " << (microphone_muted.load() ? "MUTED" : "UNMUTED") << "\n";
            return;
        }

        // Cycle compression level (Ctrl+Alt+Shift+L)
        if (is_l)
        {
            std::lock_guard<std::mutex> lock(config_mutex);
            if (!current_config.compress)
            {
                current_config.compress = 1;
                current_config.compression_level = 0;
                std::cerr << "[COMPRESS] Enabled (fast LZ4)\n";
            }
            else
            {
                current_config.compression_level = (current_config.compression_level + 3) % 15;
                if (current_config.compression_level == 0)
                {
                    current_config.compress = 0;
                    std::cerr << "[COMPRESS] Disabled\n";
                }
                else
                {
                    std::cerr << "[COMPRESS] Level: " << (int)current_config.compression_level << " (HC)\n";
                }
            }
            get_terminal_size((int &)current_config.term_width, (int &)current_config.term_height);
            send_client_config(current_config);
            return;
        }

        // === ROTATION ===

        // Rotate left by 15° (Ctrl+Alt+Shift+[)
        if (is_leftbracket)
        {
            {
                std::lock_guard<std::mutex> lock2(clear_screen_mutex);
                clear_screen_requested.store(true);
                skip_frames_counter.store(5);  // Skip 5 frames to allow server to process config
            }
            adjust_rotation(-15.0);
            return;
        }

        // Rotate right by 15° (Ctrl+Alt+Shift+])
        if (is_rightbracket)
        {
            {
                std::lock_guard<std::mutex> lock2(clear_screen_mutex);
                clear_screen_requested.store(true);
                skip_frames_counter.store(5);  // Skip 5 frames to allow server to process config
            }
            adjust_rotation(15.0);
            return;
        }

        // Reset rotation (Ctrl+Alt+Shift+\)
        if (is_backslash)
        {
            {
                std::lock_guard<std::mutex> lock2(clear_screen_mutex);
                clear_screen_requested.store(true);
                skip_frames_counter.store(5);  // Skip 5 frames to allow server to process config
            }
            set_rotation(0.0);
            return;
        }

        // Rotate 90° clockwise (Ctrl+Alt+Shift+T)
        if (is_t)
        {
            {
                std::lock_guard<std::mutex> lock2(clear_screen_mutex);
                clear_screen_requested.store(true);
                skip_frames_counter.store(5);  // Skip 5 frames to allow server to process config
            }
            adjust_rotation(90.0);
            return;
        }

        // Rotate 90° counter-clockwise (Ctrl+Alt+Shift+Y)
        if (is_y)
        {
            {
                std::lock_guard<std::mutex> lock2(clear_screen_mutex);
                clear_screen_requested.store(true);
                skip_frames_counter.store(5);  // Skip 5 frames to allow server to process config
            }
            adjust_rotation(-90.0);
            return;
        }

        // === FPS / OUTPUT ===

        // Increase FPS (Ctrl+Alt+Shift+J)
        if (is_j)
        {
            adjust_fps(5);
            return;
        }

        // Decrease FPS (Ctrl+Alt+Shift+K)
        if (is_k)
        {
            adjust_fps(-5);
            return;
        }

        // Cycle output (Ctrl+Alt+Shift+`)
        if (is_grave)
        {
            cycle_output();
            return;
        }

        // Toggle compression (Ctrl+Alt+Shift+U)
        if (is_u)
        {
            std::lock_guard<std::mutex> lock(config_mutex);
            current_config.compress = !current_config.compress;
            std::cerr << "[COMPRESS] " << (current_config.compress ? "ENABLED" : "DISABLED") << "\n";
            get_terminal_size((int &)current_config.term_width, (int &)current_config.term_height);
            send_client_config(current_config);
            return;
        }

        // === QUICK RENDERER SELECTION ===

        // Braille (Ctrl+Alt+Shift+1)
        if (is_1)
        {
            {
                std::lock_guard<std::mutex> lock2(clear_screen_mutex);
                clear_screen_requested.store(true);
                skip_frames_counter.store(5);  // Skip 5 frames to allow server to process config
            }
            set_renderer(0);
            return;
        }

        // Blocks (Ctrl+Alt+Shift+2)
        if (is_2)
        {
            {
                std::lock_guard<std::mutex> lock2(clear_screen_mutex);
                clear_screen_requested.store(true);
                skip_frames_counter.store(5);  // Skip 5 frames to allow server to process config
            }
            set_renderer(1);
            return;
        }

        // ASCII (Ctrl+Alt+Shift+3)
        if (is_3)
        {
            {
                std::lock_guard<std::mutex> lock2(clear_screen_mutex);
                clear_screen_requested.store(true);
                skip_frames_counter.store(5);  // Skip 5 frames to allow server to process config
            }
            set_renderer(2);
            return;
        }

        // Hybrid (Ctrl+Alt+Shift+4)
        if (is_4)
        {
            {
                std::lock_guard<std::mutex> lock2(clear_screen_mutex);
                clear_screen_requested.store(true);
                skip_frames_counter.store(5);  // Skip 5 frames to allow server to process config
            }
            set_renderer(3);
            return;
        }
    }

    // Only forward to server if input forwarding is enabled
    if (!feature_input || input_socket < 0 || !input_forwarding_enabled.load())
        return;

    // Send normal key event
    MessageType type = MessageType::KEY_EVENT;
    KeyEvent evt{
        keycode,
        (uint8_t)pressed,
        (uint8_t)shift_pressed.load(),
        (uint8_t)ctrl_pressed.load(),
        (uint8_t)alt_pressed.load()};

    send(input_socket, &type, sizeof(type), MSG_NOSIGNAL);
    send(input_socket, &evt, sizeof(evt), MSG_NOSIGNAL);
}

static void send_mouse_move(int x, int y)
{
    // Always update zoom center locally if enabled
    if (zoom_state.enabled.load() && zoom_state.follow_mouse.load())
    {
        zoom_state.center_x = x;
        zoom_state.center_y = y;

        // Send zoom config update to server (throttled - sent every frame anyway)
        static auto last_zoom_update = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_zoom_update).count() >= 16)
        {
            send_zoom_config();
            last_zoom_update = now;
        }
    }

    // Only forward to server if input forwarding is enabled
    if (!feature_input || input_socket < 0 || !input_forwarding_enabled.load())
        return;

    MessageType type = MessageType::MOUSE_MOVE;
    MouseMove evt{x, y, (uint32_t)screen_width.load(), (uint32_t)screen_height.load()};

    send(input_socket, &type, sizeof(type), MSG_NOSIGNAL);
    send(input_socket, &evt, sizeof(evt), MSG_NOSIGNAL);
}

static void send_mouse_button(uint32_t button, bool pressed)
{
    // Only forward to server if input forwarding is enabled
    if (!feature_input || input_socket < 0 || !input_forwarding_enabled.load())
        return;

    MessageType type = MessageType::MOUSE_BUTTON;
    MouseButton evt{button, (uint8_t)pressed};

    send(input_socket, &type, sizeof(type), MSG_NOSIGNAL);
    send(input_socket, &evt, sizeof(evt), MSG_NOSIGNAL);
}

static void send_mouse_scroll(double dx, double dy)
{
    // Only forward to server if input forwarding is enabled
    if (!feature_input || input_socket < 0 || !input_forwarding_enabled.load())
        return;

    // Send vertical scroll
    if (dy != 0)
    {
        MessageType type = MessageType::MOUSE_SCROLL;
        MouseScroll evt{dy > 0 ? 1 : -1};

        send(input_socket, &type, sizeof(type), MSG_NOSIGNAL);
        send(input_socket, &evt, sizeof(evt), MSG_NOSIGNAL);
    }
}

// libinput event processing
static void process_libinput_events()
{
    struct libinput_event *event;

    while ((event = libinput_get_event(li)) != nullptr)
    {
        enum libinput_event_type type = libinput_event_get_type(event);

        switch (type)
        {
        case LIBINPUT_EVENT_DEVICE_ADDED:
        {
            struct libinput_device *device = libinput_event_get_device(event);
            const char *name = libinput_device_get_name(device);
            std::cerr << "[HOTPLUG] Device added: " << name << "\n";

            // Log capabilities
            if (libinput_device_has_capability(device, LIBINPUT_DEVICE_CAP_KEYBOARD))
            {
                std::cerr << "[HOTPLUG]   - Keyboard capability\n";
            }
            if (libinput_device_has_capability(device, LIBINPUT_DEVICE_CAP_POINTER))
            {
                std::cerr << "[HOTPLUG]   - Pointer capability\n";
            }
            if (libinput_device_has_capability(device, LIBINPUT_DEVICE_CAP_TOUCH))
            {
                std::cerr << "[HOTPLUG]   - Touch capability\n";
            }
            break;
        }

        case LIBINPUT_EVENT_DEVICE_REMOVED:
        {
            struct libinput_device *device = libinput_event_get_device(event);
            const char *name = libinput_device_get_name(device);
            std::cerr << "[HOTPLUG] Device removed: " << name << "\n";
            break;
        }

        case LIBINPUT_EVENT_KEYBOARD_KEY:
        {
            struct libinput_event_keyboard *kb = libinput_event_get_keyboard_event(event);
            uint32_t key = libinput_event_keyboard_get_key(kb);
            enum libinput_key_state state = libinput_event_keyboard_get_key_state(kb);

            send_key_event(key, state == LIBINPUT_KEY_STATE_PRESSED);
            break;
        }

        case LIBINPUT_EVENT_POINTER_MOTION:
        {
            struct libinput_event_pointer *ptr = libinput_event_get_pointer_event(event);
            double dx = libinput_event_pointer_get_dx(ptr);
            double dy = libinput_event_pointer_get_dy(ptr);

            int new_x = current_mouse_x.load() + (int)dx;
            int new_y = current_mouse_y.load() + (int)dy;

            new_x = std::max(0, std::min(new_x, screen_width.load() - 1));
            new_y = std::max(0, std::min(new_y, screen_height.load() - 1));

            current_mouse_x = new_x;
            current_mouse_y = new_y;

            send_mouse_move(new_x, new_y);
            break;
        }

        case LIBINPUT_EVENT_POINTER_BUTTON:
        {
            struct libinput_event_pointer *ptr = libinput_event_get_pointer_event(event);
            uint32_t button = libinput_event_pointer_get_button(ptr);
            enum libinput_button_state state = libinput_event_pointer_get_button_state(ptr);

            // Convert Linux button codes to our protocol
            uint32_t btn = 1;
            if (button == BTN_RIGHT)
                btn = 3;
            else if (button == BTN_MIDDLE)
                btn = 2;

            send_mouse_button(btn, state == LIBINPUT_BUTTON_STATE_PRESSED);
            break;
        }

        case LIBINPUT_EVENT_POINTER_AXIS:
        {
            struct libinput_event_pointer *ptr = libinput_event_get_pointer_event(event);

            if (libinput_event_pointer_has_axis(ptr, LIBINPUT_POINTER_AXIS_SCROLL_VERTICAL))
            {
                double value = libinput_event_pointer_get_axis_value(ptr, LIBINPUT_POINTER_AXIS_SCROLL_VERTICAL);
                send_mouse_scroll(0, value);
            }
            break;
        }

        default:
            break;
        }

        libinput_event_destroy(event);
    }
}

// Input thread with hotplug monitoring
static void input_thread()
{
    struct pollfd fds[2];
    fds[0].fd = libinput_get_fd(li);
    fds[0].events = POLLIN;

    struct udev_monitor *mon = udev_monitor_new_from_netlink(udev, "udev");
    udev_monitor_filter_add_match_subsystem_devtype(mon, "input", NULL);
    udev_monitor_enable_receiving(mon);

    fds[1].fd = udev_monitor_get_fd(mon);
    fds[1].events = POLLIN;

    std::cerr << "[INPUT] Thread started with libinput and hotplug monitoring\n";

    while (running)
    {
        int ret = poll(fds, 2, 1); // CHANGED: 1ms instead of 100ms!

        if (ret < 0)
        {
            if (errno == EINTR)
                continue;
            std::cerr << "[INPUT] Poll error: " << strerror(errno) << "\n";
            break;
        }

        if (ret == 0)
            continue;

        // Handle libinput events
        if (fds[0].revents & POLLIN)
        {
            libinput_dispatch(li);
            process_libinput_events();
        }

        // Handle udev hotplug events
        if (fds[1].revents & POLLIN)
        {
            struct udev_device *dev = udev_monitor_receive_device(mon);
            if (dev)
            {
                const char *action = udev_device_get_action(dev);
                const char *devnode = udev_device_get_devnode(dev);

                if (action && devnode)
                {
                    if (strcmp(action, "add") == 0)
                    {
                        std::cerr << "[HOTPLUG] udev: Device added: " << devnode << "\n";
                        libinput_dispatch(li);
                        process_libinput_events();
                    }
                    else if (strcmp(action, "remove") == 0)
                    {
                        std::cerr << "[HOTPLUG] udev: Device removed: " << devnode << "\n";
                    }
                }

                udev_device_unref(dev);
            }
        }
    }

    udev_monitor_unref(mon);
    std::cerr << "[INPUT] Thread stopped\n";
}

int main(int argc, char **argv)
{
    argparse::ArgumentParser program("waytermirror_client");
    program.add_argument("-H", "--host")
        .required()
        .help("Server hostname or IP");

    program.add_argument("-P", "--port")
        .default_value(9999)
        .scan<'i', int>()
        .help("Server port");

    program.add_argument("-o", "--output")
        .default_value(std::string("0"))
        .help("Output index (0, 1, 2...) or 'follow'");

    program.add_argument("-F", "--fps")
        .default_value(30)
        .scan<'i', int>()
        .help("Desired FPS");

    program.add_argument("-M", "--mode")
        .default_value(std::string("256"))
        .help("Color mode: 16, 256, true");

    program.add_argument("-R", "--renderer")
        .default_value(std::string("braille"))
        .help("Renderer");

    program.add_argument("-S", "--scale")
        .default_value(1.0)
        .scan<'g', double>()
        .help("Scale");

    program.add_argument("-k", "--keep-aspect-ratio")
        .default_value(false)
        .implicit_value(true)
        .help("Keep aspect ratio");

    program.add_argument("-c", "--compress")
        .default_value(false)
        .implicit_value(true)
        .help("Enable compression");

    program.add_argument("-L", "--compression-level")
        .default_value(0)
        .scan<'i', int>()
        .help("LZ4 compression level: 0=fast, 1–12=HC");

    program.add_argument("-C", "--center-mouse")
        .default_value(false)
        .implicit_value(true)
        .help("Start mouse at screen center");

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

    program.add_argument("-x", "--exclusive-input")
        .default_value(false)
        .implicit_value(true)
        .help("Exclusive access to input devices");

    program.add_argument("-p", "--no-microphone")
        .default_value(false)
        .implicit_value(true)
        .help("Disable microphone forwarding");

    program.add_argument("-d", "--detail-level")
        .default_value(50)
        .scan<'i', int>()
        .help("Detail level: 0=smooth, 100=sharp");

    program.add_argument("-z", "--zoom")
        .default_value(false)
        .implicit_value(true)
        .help("Start with zoom enabled");

    program.add_argument("-Z", "--zoom-level")
        .default_value(2.0)
        .scan<'g', double>()
        .help("Zoom magnification (1.0-10.0)");

    program.add_argument("-X", "--zoom-width")
        .default_value(800)
        .scan<'i', int>()
        .help("Zoom viewport width in pixels");

    program.add_argument("-Y", "--zoom-height")
        .default_value(600)
        .scan<'i', int>()
        .help("Zoom viewport height in pixels");

    program.add_argument("-f", "--zoom-follow")
        .default_value(true)
        .implicit_value(true)
        .help("Zoom follows mouse cursor");

    program.add_argument("-s", "--zoom-smooth")
        .default_value(true)
        .implicit_value(true)
        .help("Smooth zoom panning");

    program.add_argument("-D", "--zoom-speed")
        .default_value(20)
        .scan<'i', int>()
        .help("Zoom pan speed (pixels per frame)");

    program.add_argument("-r", "--render-device")
        .default_value(std::string("cpu"))
        .help("Rendering device: cpu, cuda");

    program.add_argument("-Q", "--quality")
        .default_value(50)
        .scan<'i', int>()
        .help("Rendering quality: 0=fast/poor, 50=balanced, 100=slow/best");

    program.add_argument("-T", "--rotation")
        .default_value(0.0)
        .scan<'g', double>()
        .help("Rotation angle (degrees): 0...360");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (...)
    {
        std::cerr << program;
        return 1;
    }

    // Parse feature flags
    feature_video = !program.get<bool>("--no-video");
    feature_audio = !program.get<bool>("--no-audio");
    feature_input = !program.get<bool>("--no-input");
    feature_microphone = !program.get<bool>("--no-microphone");

    exclusive_mode = program.get<bool>("--exclusive-input");

    std::cerr << "=== Client Features ===\n";
    std::cerr << "Video: " << (feature_video ? "ON" : "OFF") << "\n";
    std::cerr << "Audio: " << (feature_audio ? "ON" : "OFF") << "\n";
    std::cerr << "Input: " << (feature_input ? exclusive_mode ? "ON (Exclusive)" : "ON (Shared)" : "OFF") << "\n";
    std::cerr << "Microphone: " << (feature_microphone ? "ON" : "OFF") << "\n";
    std::cerr << "=======================\n\n";

    // Always initialize libinput for local shortcuts (exit combo, zoom toggle)
    // Input forwarding to server is controlled by feature_input flag
    bool libinput_ok = false;
    {
        // Check if running with sufficient privileges for libinput
        if (geteuid() != 0)
        {
            std::cerr << "\n=== WARNING ===\n";
            std::cerr << "Not running as root. Input devices may not be accessible.\n";
            std::cerr << "Run with sudo or add your user to the 'input' group:\n";
            std::cerr << "  sudo usermod -a -G input $USER\n";
            std::cerr << "Then log out and back in.\n";
            std::cerr << "===============\n\n";
        }

        // Initialize libinput
        udev = udev_new();
        if (!udev)
        {
            std::cerr << "Failed to initialize udev\n";
            std::cerr << "Local shortcuts (exit combo, zoom) will not work\n";
        }
        else
        {
            li = libinput_udev_create_context(&interface, nullptr, udev);
            if (!li)
            {
                std::cerr << "Failed to initialize libinput\n";
                std::cerr << "Local shortcuts (exit combo, zoom) will not work\n";
                udev_unref(udev);
                udev = nullptr;
            }
            else
            {
                if (libinput_udev_assign_seat(li, "seat0") != 0)
                {
                    std::cerr << "Failed to assign seat to libinput\n";
                    std::cerr << "Local shortcuts (exit combo, zoom) will not work\n";
                    libinput_unref(li);
                    li = nullptr;
                    udev_unref(udev);
                    udev = nullptr;
                }
                else
                {
                    std::cerr << "[INPUT] libinput initialized successfully\n";
                    std::cerr << "[INPUT] Hotplug support enabled\n";

                    // Enumerate initial devices
                    std::cerr << "[INPUT] Initial devices:\n";
                    libinput_dispatch(li);
                    process_libinput_events();
                    libinput_ok = true;
                }
            }
        }
    }

    // Network setup
    std::string session_id = generate_uuid();
    std::string host = program.get<std::string>("--host");
    int port = program.get<int>("--port");

    std::cerr << "[INIT] Session ID: " << session_id << "\n";

    // Connect config socket (ALWAYS, regardless of features)
    config_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (config_socket < 0)
    {
        std::cerr << "Failed to create config socket\n";
        return 1;
    }

    sockaddr_in config_addr{};
    config_addr.sin_family = AF_INET;
    config_addr.sin_port = htons(port + 3); // Port+3 for config
    if (inet_pton(AF_INET, host.c_str(), &config_addr.sin_addr) <= 0)
    {
        std::cerr << "Invalid host address\n";
        return 1;
    }

    if (connect(config_socket, (sockaddr *)&config_addr, sizeof(config_addr)) < 0)
    {
        std::cerr << "Failed to connect config socket to " << host << ":" << (port + 3) << "\n";
        return 1;
    }

    if (!send_session_id(config_socket, session_id))
    {
        std::cerr << "Failed to send session ID to config socket\n";
        return 1;
    }

    std::cerr << "[CONFIG] Connected to " << host << ":" << (port + 3) << "\n";

    // Connect frame socket (only if video enabled)
    if (feature_video)
    {
        frame_socket = socket(AF_INET, SOCK_STREAM, 0);
        if (frame_socket < 0)
        {
            std::cerr << "Failed to create frame socket\n";
            return 1;
        }

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

        if (connect(frame_socket, (sockaddr *)&addr, sizeof(addr)) < 0)
        {
            std::cerr << "Failed to connect frame socket to " << host << ":" << port << "\n";
            return 1;
        }

        int nodelay = 1;
        setsockopt(frame_socket, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

        if (!send_session_id(frame_socket, session_id))
        {
            std::cerr << "Failed to send session ID to frame socket\n";
            return 1;
        }

        std::cerr << "[VIDEO] Connected to " << host << ":" << port << "\n";
    }

    // Connect input socket (only if input enabled)
    if (feature_input)
    {
        input_socket = socket(AF_INET, SOCK_STREAM, 0);
        if (input_socket < 0)
        {
            std::cerr << "Failed to create input socket\n";
            if (!feature_video && !feature_audio && !feature_microphone)
                return 1;
            feature_input = false;
        }
        else
        {
            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port + 1);
            inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

            if (connect(input_socket, (sockaddr *)&addr, sizeof(addr)) < 0)
            {
                std::cerr << "Failed to connect input socket\n";
                close(input_socket);
                input_socket = -1;
                if (!feature_video && !feature_audio && !feature_microphone)
                    return 1;
                feature_input = false;
            }
            else
            {
                int nodelay = 1;
                setsockopt(input_socket, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

                if (!send_session_id(input_socket, session_id))
                {
                    std::cerr << "Failed to send session ID to input socket\n";
                    close(input_socket);
                    input_socket = -1;
                    feature_input = false;
                }
                else
                {
                    std::cerr << "[INPUT] Connected to " << host << ":" << (port + 1) << "\n";
                }
            }
        }
    }

    // Connect audio socket (only if audio enabled)
    if (feature_audio)
    {
        audio_socket = socket(AF_INET, SOCK_STREAM, 0);
        if (audio_socket < 0)
        {
            std::cerr << "Warning: Failed to create audio socket\n";
            feature_audio = false;
        }
        else
        {
            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port + 2);
            inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

            if (connect(audio_socket, (sockaddr *)&addr, sizeof(addr)) < 0)
            {
                std::cerr << "Warning: Failed to connect audio socket\n";
                close(audio_socket);
                audio_socket = -1;
                feature_audio = false;
            }
            else
            {
                int nodelay = 1;
                setsockopt(audio_socket, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

                if (!send_session_id(audio_socket, session_id))
                {
                    std::cerr << "Warning: Failed to send session ID to audio socket\n";
                    close(audio_socket);
                    audio_socket = -1;
                    feature_audio = false;
                }
                else
                {
                    std::cerr << "[AUDIO IN] Connected to " << host << ":" << (port + 2) << "\n";
                }
            }
        }
    }

    // Connect microphone socket (only if microphone enabled)
    std::thread microphone_thr;
    if (feature_microphone)
    {
        microphone_socket = socket(AF_INET, SOCK_STREAM, 0);
        if (microphone_socket < 0)
        {
            std::cerr << "Warning: Failed to create microphone socket\n";
            feature_microphone = false;
        }
        else
        {
            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port + 4);
            inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

            if (connect(microphone_socket, (sockaddr *)&addr, sizeof(addr)) < 0)
            {
                std::cerr << "Warning: Failed to connect microphone socket to " << host << ":" << (port + 4) << "\n";
                close(microphone_socket);
                microphone_socket = -1;
                feature_microphone = false;
            }
            else
            {
                int nodelay = 1;
                setsockopt(microphone_socket, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

                if (!send_session_id(microphone_socket, session_id))
                {
                    std::cerr << "Warning: Failed to send session ID to microphone socket\n";
                    close(microphone_socket);
                    microphone_socket = -1;
                    feature_microphone = false;
                }
                else
                {
                    std::cerr << "[MICROPHONE OUT] Connected to " << host << ":" << (port + 4) << "\n";
                    // Start microphone send thread
                    microphone_thr = std::thread(microphone_send_thread);
                }
            }
        }
    }

    // Check that at least config socket is working
    if (config_socket < 0)
    {
        std::cerr << "Error: Config connection failed\n";
        return 1;
    }

    std::string output_arg = program.get<std::string>("--output");
    std::string mode_str = program.get<std::string>("--mode");
    std::string renderer = program.get<std::string>("--renderer");

    // Send initial config through config socket
    {
        std::lock_guard<std::mutex> lock(config_mutex);
        current_config.output_index = (output_arg == "follow") ? 0xFFFFFFFF : std::stoi(output_arg);
        current_config.follow_focus = (output_arg == "follow") ? 1 : 0;
        current_config.fps = program.get<int>("--fps");
        get_terminal_size((int &)current_config.term_width, (int &)current_config.term_height);
        current_config.color_mode = (mode_str == "16") ? 0 : (mode_str == "256") ? 1
                                                                                 : 2;
        current_config.renderer = (renderer == "braille") ? 0 : (renderer == "blocks") ? 1
                                                            : (renderer == "ascii")    ? 2
                                                                                       : 3;
        current_config.keep_aspect_ratio = program.get<bool>("--keep-aspect-ratio") ? 1 : 0;
        current_config.scale_factor = program.get<double>("--scale");
        current_config.compress = program.get<bool>("--compress") ? 1 : 0;
        current_config.compression_level = program.get<int>("--compression-level");
        current_config.detail_level = std::clamp(program.get<int>("--detail-level"), 0, 100);
        current_config.render_device = (program.get<std::string>("--render-device") == "cuda") ? 1 : 0;
        current_config.quality = std::clamp(program.get<int>("--quality"), 0, 100);
        current_config.rotation_angle = std::fmod(program.get<double>("--rotation"), 360.0);
        if (current_config.rotation_angle < 0)
        {
            current_config.rotation_angle += 360.0;
        }
    }

    zoom_state.enabled = program.get<bool>("--zoom");
    zoom_state.zoom_level = std::clamp(program.get<double>("--zoom-level"), 1.0, 10.0);
    zoom_state.view_width = program.get<int>("--zoom-width");
    zoom_state.view_height = program.get<int>("--zoom-height");
    zoom_state.follow_mouse = program.get<bool>("--zoom-follow");
    zoom_state.smooth_pan = program.get<bool>("--zoom-smooth");
    zoom_state.pan_speed = std::clamp(program.get<int>("--zoom-speed"), 1, 100);
    zoom_state.center_x = screen_width.load() / 2;
    zoom_state.center_y = screen_height.load() / 2;

    std::cerr << "=== Zoom Configuration ===\n";
    std::cerr << "Enabled: " << (zoom_state.enabled.load() ? "YES" : "NO") << "\n";
    std::cerr << "Level: " << zoom_state.zoom_level.load() << "x\n";
    std::cerr << "Viewport: " << zoom_state.view_width.load() << "x" << zoom_state.view_height.load() << "\n";
    std::cerr << "Follow Mouse: " << (zoom_state.follow_mouse.load() ? "YES" : "NO") << "\n";
    std::cerr << "Smooth Pan: " << (zoom_state.smooth_pan.load() ? "YES" : "NO") << "\n";
    std::cerr << "==========================\n\n";

    std::cerr << "=== Rendering Settings ===\n";
    std::cerr << "Detail Level: " << (int)current_config.detail_level << " (0=smooth, 100=sharp)\n";
    std::cerr << "Quality: " << (int)current_config.quality << " (0=fast, 100=best)\n";
    std::cerr << "Render Device: " << (current_config.render_device == 1 ? "CUDA" : "CPU") << "\n";
    std::cerr << "===========================\n\n";

    send_client_config(current_config);
    if (zoom_state.enabled.load())
    {
        send_zoom_config();
    }

    // Wait for screen info (only if video enabled)
    if (feature_video)
    {
        std::cerr << "[INIT] Waiting for screen info from server...\n";
        for (int i = 0; i < 20 && screen_width.load() == 1920; i++)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

            struct pollfd pfd = {frame_socket, POLLIN, 0};
            if (poll(&pfd, 1, 0) > 0)
            {
                MessageType type;
                if (recv(frame_socket, &type, sizeof(type), MSG_PEEK) == sizeof(type))
                {
                    if (type == MessageType::SCREEN_INFO)
                    {
                        recv(frame_socket, &type, sizeof(type), 0);
                        ScreenInfo info;
                        if (recv(frame_socket, &info, sizeof(info), 0) == sizeof(info))
                        {
                            screen_width = info.width;
                            screen_height = info.height;
                            std::cerr << "[INIT] Received screen info: " << info.width << "x" << info.height << "\n";
                            break;
                        }
                    }
                }
            }
        }

        if (screen_width.load() == 1920)
        {
            std::cerr << "[INIT] Warning: No screen info received, using default 1920x1080\n";
        }
    }

    // Initialize mouse position
    if (feature_input && program.get<bool>("--center-mouse"))
    {
        current_mouse_x = screen_width.load() / 2;
        current_mouse_y = screen_height.load() / 2;
        std::cerr << "[INIT] Mouse centered at " << current_mouse_x.load()
                  << "," << current_mouse_y.load() << "\n";
    }

    // Start threads based on enabled features
    // Input thread always runs for local shortcuts (exit combo, zoom toggle)
    std::thread input_thr;
    if (libinput_ok)
    {
        input_thr = std::thread(input_thread);
    }

    std::thread audio_thr;
    if (feature_audio)
    {
        audio_thr = std::thread(audio_receive_thread);
    }

    // Setup terminal (only if video enabled)
    if (feature_video)
    {
        std::cout << "\033[?25l\033[?47h\033[2J\033[H\033[0m" << std::flush;
    }

    std::cerr << "\n=== Waytermirror Client Started ===\n";
    std::cerr << "Session: " << session_id << "\n";
    if (feature_video)
    {
        std::cerr << "Output: " << program.get<std::string>("--output") << " | Resolution: "
                  << screen_width.load() << "x" << screen_height.load() << "\n";
        std::cerr << "Renderer: " << renderer << " | Color mode: " << mode_str << "\n";
        std::cerr << "FPS: " << current_config.fps << " | Scale: " << current_config.scale_factor
                  << (current_config.keep_aspect_ratio ? " (Auto-fit)" : "") << "\n";
        std::cerr << "Compression: " << (current_config.compress ? "ON" : "OFF") << "\n";
    }
    if (libinput_ok)
    {
        std::cerr << "\n=== Keyboard Shortcuts (Ctrl+Alt+Shift+Key) ===\n";
        std::cerr << "  Q=Quit  H=Help  I=Toggle Input  G=Toggle Grab\n";
        std::cerr << "  Z=Zoom  +/-=Zoom In/Out  0=Reset Zoom  Arrows=Pan\n";
        std::cerr << "  R=Renderer  C=Color  D/S=Detail  W/E=Quality\n";
        std::cerr << "  P=Pause  A=Audio Mute  M=Mic Mute  F=Focus-Follow\n";
        if (feature_input)
        {
            std::cerr << "Input forwarding: ENABLED\n";
        }
        else
        {
            std::cerr << "Input forwarding: DISABLED (local shortcuts only)\n";
        }
        std::cerr << "Press Ctrl+Alt+Shift+H for full shortcut list\n";
    }
    if (feature_audio)
    {
        std::cerr << "Audio playback enabled (system audio from server)\n";
    }
    if (feature_microphone)
    {
        std::cerr << "Microphone capture enabled (sending to server)\n";
    }
    std::cerr << "================================================\n\n";

    // Main loop
    while (running)
    {
        if (feature_video)
        {
            std::string rendered;
            int new_term_width, new_term_height;
            get_terminal_size(new_term_width, new_term_height);

            // Handle terminal resize
            static int last_term_width = new_term_width;
            static int last_term_height = new_term_height;

            if (new_term_width != last_term_width || new_term_height != last_term_height)
            {
                last_term_width = new_term_width;
                last_term_height = new_term_height;

                std::lock_guard<std::mutex> lock(config_mutex);
                current_config.term_width = new_term_width;
                current_config.term_height = new_term_height;
                send_client_config(current_config);

                std::cout << "\033[2J";
                std::cerr << "[RESIZE] Terminal resized to " << new_term_width << "x"
                          << new_term_height << "\n";
            }

            // Handle clear screen request
            {
                std::lock_guard<std::mutex> lock(clear_screen_mutex);
                if (clear_screen_requested.load()) 
                {
                    clear_screen_requested.store(false);
                    std::cout << "\033[H\033[2J" << std::flush;
                    std::cerr << "[CLEAR] Screen cleared as requested\n";
                }
            }

            // Skip frames if counter is active (waiting for server to process config changes)
            int skip_count = skip_frames_counter.load();
            if (skip_count > 0)
            {
                skip_frames_counter.store(skip_count - 1);
                // Still consume frames to prevent buffer buildup
                std::string dummy;
                receive_newest_frame(dummy);
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
            }
            else if (!video_paused.load() && receive_newest_frame(rendered))
            {
                std::cout << "\033[H" << rendered << std::flush;
            }
            else if (video_paused.load())
            {
                // Still consume frames to prevent buffer buildup, but don't display
                receive_newest_frame(rendered);
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
            }
            else
            {
                // No frame available, sleep briefly to avoid busy-looping
                std::this_thread::sleep_for(std::chrono::microseconds(1000));
            }
        }
        else
        {
            // No video - just wait for exit combo or Ctrl+C
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    std::cerr << "\n[EXIT] Shutting down...\n";

    // Cleanup terminal (only if video enabled)
    if (feature_video)
    {
        std::cout << "\033[0m\033[2J\033[H\033[?47l\033[?25h" << std::flush;
    }

    running = false;

    // Join threads in order
    if (feature_microphone && microphone_thr.joinable())
    {
        microphone_thr.join();
    }

    // Input thread runs if libinput initialized (for local shortcuts)
    if (input_thr.joinable())
    {
        input_thr.join();
    }

    if (feature_audio && audio_thr.joinable())
    {
        audio_thr.join();
    }

    // Close sockets
    if (frame_socket >= 0)
        close(frame_socket);
    if (input_socket >= 0)
        close(input_socket);
    if (audio_socket >= 0)
        close(audio_socket);
    if (config_socket >= 0)
        close(config_socket);
    if (microphone_socket >= 0)
        close(microphone_socket);

    // Cleanup resources
    if (li)
    {
        libinput_unref(li);
    }
    if (udev)
    {
        udev_unref(udev);
    }

    if (feature_audio)
    {
        cleanup_audio_playback();
    }

    if (feature_microphone)
    {
        cleanup_microphone_capture();
    }

    std::cerr << "[EXIT] Shutdown complete.\n";
    return 0;
}
