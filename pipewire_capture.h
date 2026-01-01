#pragma once

#include <pipewire/pipewire.h>
#include <spa/param/video/format-utils.h>
#include <spa/param/props.h>
#include <spa/debug/types.h>
#include <vector>
#include <mutex>
#include <atomic>
#include <memory>
#include <functional>
#include <string>
#include <gio/gio.h>

struct PipeWireCapture {
    pw_thread_loop *loop = nullptr;
    pw_stream *stream = nullptr;
    pw_context *context = nullptr;
    
    std::vector<uint8_t> latest_frame;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t stride = 0;
    uint32_t format = 0;
    
    std::mutex mutex;
    std::atomic<bool> frame_ready{false};
    std::atomic<bool> running{true};
    
    std::function<void()> on_frame_callback;
    
    // Portal integration (keep session alive!)
    GDBusConnection *portal_connection = nullptr;
    std::string session_handle;
    uint32_t pipewire_node_id = 0;
    std::atomic<bool> portal_ready{false};
    
    bool init(uint32_t output_index);
    void cleanup();
    bool get_frame(std::vector<uint8_t> &out_frame, uint32_t &out_width, 
                   uint32_t &out_height, uint32_t &out_stride);
    
private:
    bool request_screen_cast();
};

bool pipewire_capture_available();
