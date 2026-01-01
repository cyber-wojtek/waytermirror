#include "pipewire_capture.h"
#include <iostream>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

static void on_pipewire_process(void *userdata) {
    PipeWireCapture *cap = static_cast<PipeWireCapture*>(userdata);
    
    pw_buffer *b = pw_stream_dequeue_buffer(cap->stream);
    if (!b) {
        std::cerr << "[PW] No buffer available\n";
        return;
    }
    
    spa_buffer *buf = b->buffer;
    if (!buf->datas[0].data) {
        std::cerr << "[PW] No data in buffer\n";
        pw_stream_queue_buffer(cap->stream, b);
        return;
    }
    
    uint8_t *src = static_cast<uint8_t*>(buf->datas[0].data);
    uint32_t size = buf->datas[0].chunk->size;
    
    if (size == 0) {
        pw_stream_queue_buffer(cap->stream, b);
        return;
    }
    
    {
        std::lock_guard<std::mutex> lock(cap->mutex);
        
        // Copy frame data
        cap->latest_frame.resize(size);
        memcpy(cap->latest_frame.data(), src, size);
        cap->frame_ready = true;
    }
    
    // Notify callback
    if (cap->on_frame_callback) {
        cap->on_frame_callback();
    }
    
    pw_stream_queue_buffer(cap->stream, b);
}

static void on_pipewire_param_changed(void *userdata, uint32_t id, const spa_pod *param) {
    PipeWireCapture *cap = static_cast<PipeWireCapture*>(userdata);
    
    if (!param || id != SPA_PARAM_Format) {
        return;
    }
    
    spa_video_info_raw info;
    if (spa_format_video_raw_parse(param, &info) < 0) {
        std::cerr << "[PW] Failed to parse video format\n";
        return;
    }
    
    std::lock_guard<std::mutex> lock(cap->mutex);
    cap->width = info.size.width;
    cap->height = info.size.height;
    cap->stride = info.size.width * 4; // Assume BGRA
    cap->format = info.format;
    
    std::cerr << "[PW] Format: " << cap->width << "x" << cap->height 
              << " stride=" << cap->stride << " format=" << cap->format << "\n";
}

static const pw_stream_events pipewire_stream_events = {
    .version = PW_VERSION_STREAM_EVENTS,
    .param_changed = on_pipewire_param_changed,
    .process = on_pipewire_process,
};

bool PipeWireCapture::init(uint32_t output_index) {
    pw_init(nullptr, nullptr);
    
    loop = pw_thread_loop_new("pipewire-capture", nullptr);
    if (!loop) {
        std::cerr << "[PW] Failed to create thread loop\n";
        return false;
    }
    
    pw_thread_loop_lock(loop);
    
    context = pw_context_new(pw_thread_loop_get_loop(loop), nullptr, 0);
    if (!context) {
        std::cerr << "[PW] Failed to create context\n";
        pw_thread_loop_unlock(loop);
        return false;
    }
    
    // Create stream for screen capture
    stream = pw_stream_new_simple(
        pw_thread_loop_get_loop(loop),
        "waytermirror-screencapture",
        pw_properties_new(
            PW_KEY_MEDIA_TYPE, "Video",
            PW_KEY_MEDIA_CATEGORY, "Capture",
            PW_KEY_MEDIA_ROLE, "Screen",
            nullptr),
        &pipewire_stream_events,
        this);
    
    if (!stream) {
        std::cerr << "[PW] Failed to create stream\n";
        pw_thread_loop_unlock(loop);
        return false;
    }
    
    // Request BGRA format (most common for screen capture)
    uint8_t buffer[1024];
    spa_pod_builder b = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));
    
    const spa_pod *params[1];
    spa_rectangle size = SPA_RECTANGLE(1920, 1080); // Will be updated by compositor
    spa_fraction framerate = SPA_FRACTION(30, 1);
    
    auto spa_video_info_raw = SPA_VIDEO_INFO_RAW_INIT(
        .format = SPA_VIDEO_FORMAT_BGRx, // Try BGRx first, fallback to BGRA
        .size = size,
        .framerate = framerate);
    
    params[0] = spa_format_video_raw_build(&b, SPA_PARAM_EnumFormat, &spa_video_info_raw);
    
    // Connect to PipeWire
    int res = pw_stream_connect(stream,
                                PW_DIRECTION_INPUT,
                                PW_ID_ANY,
                                static_cast<pw_stream_flags>(
                                    PW_STREAM_FLAG_AUTOCONNECT |
                                    PW_STREAM_FLAG_MAP_BUFFERS),
                                params, 1);
    
    if (res < 0) {
        std::cerr << "[PW] Failed to connect stream: " << strerror(-res) << "\n";
        pw_thread_loop_unlock(loop);
        return false;
    }
    
    pw_thread_loop_unlock(loop);
    pw_thread_loop_start(loop);
    
    std::cerr << "[PW] Screen capture initialized for output " << output_index << "\n";
    return true;
}

void PipeWireCapture::cleanup() {
    running = false;
    
    if (stream) {
        pw_stream_destroy(stream);
        stream = nullptr;
    }
    
    if (context) {
        pw_context_destroy(context);
        context = nullptr;
    }
    
    if (loop) {
        pw_thread_loop_stop(loop);
        pw_thread_loop_destroy(loop);
        loop = nullptr;
    }
    
    pw_deinit();
    std::cerr << "[PW] Capture cleaned up\n";
}

bool PipeWireCapture::get_frame(std::vector<uint8_t> &out_frame, uint32_t &out_width,
                                uint32_t &out_height, uint32_t &out_stride) {
    std::lock_guard<std::mutex> lock(mutex);
    
    if (!frame_ready || latest_frame.empty()) {
        return false;
    }
    
    out_frame = latest_frame;
    out_width = width;
    out_height = height;
    out_stride = stride;
    
    return true;
}

bool pipewire_capture_available() {
    // Try to initialize PipeWire to check availability
    pw_init(nullptr, nullptr);
    bool available = true;
    
    // Just check if we can create a basic loop
    pw_thread_loop *test_loop = pw_thread_loop_new("test", nullptr);
    if (!test_loop) {
        available = false;
    } else {
        pw_thread_loop_destroy(test_loop);
    }
    
    pw_deinit();
    return available;
}
