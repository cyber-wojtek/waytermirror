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
    
    bool init(uint32_t output_index);
    void cleanup();
    bool get_frame(std::vector<uint8_t> &out_frame, uint32_t &out_width, 
                   uint32_t &out_height, uint32_t &out_stride);
};

bool pipewire_capture_available();
