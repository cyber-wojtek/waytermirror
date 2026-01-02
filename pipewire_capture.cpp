#include "pipewire_capture.h"
#include <iostream>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <gio/gio.h>
#include <glib.h>
#include <random>
#include <sstream>
#include <iomanip>

static void on_pipewire_param_changed(void *userdata, uint32_t id, const spa_pod *param) {
    PipeWireCapture *cap = static_cast<PipeWireCapture*>(userdata);
    
    // Only process Format params, ignore everything else
    if (id != SPA_PARAM_Format || !param) {
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
    
    uint32_t bytes_per_pixel = 4;
    switch (info.format) {
        case SPA_VIDEO_FORMAT_BGRx:
        case SPA_VIDEO_FORMAT_BGRA:
        case SPA_VIDEO_FORMAT_RGBx:
        case SPA_VIDEO_FORMAT_RGBA:
            bytes_per_pixel = 4;
            break;
        case SPA_VIDEO_FORMAT_RGB:
        case SPA_VIDEO_FORMAT_BGR:
            bytes_per_pixel = 3;
            break;
        default:
            bytes_per_pixel = 4;
    }
    
    cap->stride = info.size.width * bytes_per_pixel;
    cap->format = info.format;
    
    std::cerr << "[PW] Format negotiated: " << cap->width << "x" << cap->height 
              << " stride=" << cap->stride << " format=" << info.format << "\n";
}

static void on_pipewire_process(void *userdata) {
    PipeWireCapture *cap = static_cast<PipeWireCapture*>(userdata);
    
    static bool first_frame = true;
    if (first_frame) {
        std::cerr << "[PW] *** FIRST FRAME CALLBACK INVOKED ***\n";
        first_frame = false;
    }
    
    pw_buffer *b = pw_stream_dequeue_buffer(cap->stream);
    if (!b) {
        static int no_buf_count = 0;
        if (++no_buf_count % 100 == 0) {
            std::cerr << "[PW] No buffer available in process callback (" << no_buf_count << " times)\n";
        }
        return;
    }
    
    spa_buffer *buf = b->buffer;
    if (!buf->datas || !buf->datas[0].data) {
        std::cerr << "[PW] Buffer has no data pointer\n";
        pw_stream_queue_buffer(cap->stream, b);
        return;
    }
    
    uint8_t *src = static_cast<uint8_t*>(buf->datas[0].data);
    uint32_t size = buf->datas[0].chunk->size;
    
    if (size == 0) {
        static int zero_size_count = 0;
        if (++zero_size_count % 100 == 0) {
            std::cerr << "[PW] Received buffer with zero size (" << zero_size_count << " times)\n";
        }
        pw_stream_queue_buffer(cap->stream, b);
        return;
    }
    
    {
        std::lock_guard<std::mutex> lock(cap->mutex);
        
        cap->latest_frame.resize(size);
        memcpy(cap->latest_frame.data(), src, size);
        cap->frame_ready = true;
        
        static int frame_count = 0;
        if (++frame_count == 1) {
            std::cerr << "[PW] *** FIRST FRAME RECEIVED! Size=" << size << " bytes ***\n";
        }
        if (frame_count % 60 == 0) {
            std::cerr << "[PW] Frames received: " << frame_count << " (size=" << size << ")\n";
        }
    }
    
    if (cap->on_frame_callback) {
        cap->on_frame_callback();
    }
    
    pw_stream_queue_buffer(cap->stream, b);
}

static void on_pipewire_state_changed(void *userdata, enum pw_stream_state old, 
                                       enum pw_stream_state state, const char *error) {
    auto state_name = [](enum pw_stream_state s) -> const char* {
        switch(s) {
            case PW_STREAM_STATE_ERROR: return "error";
            case PW_STREAM_STATE_UNCONNECTED: return "unconnected";
            case PW_STREAM_STATE_CONNECTING: return "connecting";
            case PW_STREAM_STATE_PAUSED: return "paused";
            case PW_STREAM_STATE_STREAMING: return "streaming";
            default: return "unknown";
        }
    };
    
    std::cerr << "[PW] Stream state: " << state_name(old) << " -> " << state_name(state);
    if (error) {
        std::cerr << " (error: " << error << ")";
    }
    std::cerr << "\n";
    
    if (state == PW_STREAM_STATE_PAUSED) {
        std::cerr << "[PW] Stream PAUSED - ready to receive frames\n";
    } else if (state == PW_STREAM_STATE_STREAMING) {
        std::cerr << "[PW] Stream STREAMING\n";
    }
}

static const pw_stream_events pipewire_stream_events = {
    .version = PW_VERSION_STREAM_EVENTS,
    .state_changed = on_pipewire_state_changed,
    .param_changed = on_pipewire_param_changed,
    .process = on_pipewire_process,
};

// Generate random token for D-Bus requests
static std::string generate_token() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    
    std::ostringstream ss;
    ss << "wayterm_";
    
    // Add timestamp to ensure uniqueness
    auto now = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    ss << std::hex << ms << "_";
    
    for (int i = 0; i < 16; i++) {
        ss << std::hex << dis(gen);
    }
    return ss.str();
}

bool PipeWireCapture::request_screen_cast() {
    GError *error = nullptr;
    
    // Each capture needs its own independent D-Bus connection
    GDBusConnection *connection = g_bus_get_sync(G_BUS_TYPE_SESSION, nullptr, &error);
    if (error) {
        std::cerr << "[PORTAL] Failed to connect to D-Bus: " << error->message << "\n";
        g_error_free(error);
        return false;
    }
    
    std::string sender_name = g_dbus_connection_get_unique_name(connection);
    sender_name = sender_name.substr(1);
    for (char &c : sender_name) {
        if (c == '.') c = '_';
    }
    
    // Use unique tokens with timestamp to prevent collisions
    std::string session_token = generate_token();
    std::string request_token = generate_token();
    session_handle = "/org/freedesktop/portal/desktop/session/" + sender_name + "/" + session_token;
    std::string request_path = "/org/freedesktop/portal/desktop/request/" + sender_name + "/" + request_token;
    
    std::cerr << "[PORTAL] Creating independent ScreenCast session (token=" << session_token.substr(0, 16) << "...)\n";
    
    // Step 1: CreateSession - wait for async Response
    struct SessionData {
        std::atomic<bool> done{false};
        std::atomic<bool> success{false};
        std::string session_handle;
    };
    
    SessionData session_data;
    
    auto session_response_handler = +[](GDBusConnection *conn, const gchar *sender,
                                        const gchar *object_path, const gchar *interface,
                                        const gchar *signal_name, GVariant *parameters,
                                        gpointer user_data) {
        auto *data = static_cast<SessionData*>(user_data);
        
        std::cerr << "[PORTAL] CreateSession Response signal received\n";
        std::cerr << "[PORTAL]   Sender: " << (sender ? sender : "null") << "\n";
        std::cerr << "[PORTAL]   Object path: " << (object_path ? object_path : "null") << "\n";
        
        guint32 response;
        GVariant *results;
        g_variant_get(parameters, "(u@a{sv})", &response, &results);
        
        std::cerr << "[PORTAL]   Response code: " << response;
        const char *response_meaning[] = {"Success", "User cancelled", "Other error"};
        if (response <= 2) {
            std::cerr << " (" << response_meaning[response] << ")";
        }
        std::cerr << "\n";
        
        // Debug: print all keys in results
        std::cerr << "[PORTAL]   Results dict contains:\n";
        GVariantIter iter;
        g_variant_iter_init(&iter, results);
        const gchar *key;
        GVariant *value;
        while (g_variant_iter_next(&iter, "{&sv}", &key, &value)) {
            gchar *value_str = g_variant_print(value, TRUE);
            std::cerr << "[PORTAL]     " << key << " = " << value_str << "\n";
            g_free(value_str);
            g_variant_unref(value);
        }
        
        if (response == 0) {
            // Extract the session_handle from the results
            GVariant *handle_variant = g_variant_lookup_value(results, "session_handle", G_VARIANT_TYPE_STRING);
            if (handle_variant) {
                const char *handle = g_variant_get_string(handle_variant, nullptr);
                data->session_handle = handle;
                data->success = true;
                std::cerr << "[PORTAL] Session created successfully: " << handle << "\n";
                g_variant_unref(handle_variant);
            } else {
                std::cerr << "[PORTAL] Session created but no handle returned\n";
            }
        } else {
            std::cerr << "[PORTAL] Session creation failed with code " << response << "\n";
        }
        
        g_variant_unref(results);
        data->done = true;
    };
    
    guint session_signal_id = g_dbus_connection_signal_subscribe(
        connection,
        "org.freedesktop.portal.Desktop",
        "org.freedesktop.portal.Request",
        "Response",
        request_path.c_str(),
        nullptr,
        G_DBUS_SIGNAL_FLAGS_NONE,
        session_response_handler,
        &session_data,
        nullptr);
    
    GVariantBuilder session_options;
    g_variant_builder_init(&session_options, G_VARIANT_TYPE_VARDICT);
    g_variant_builder_add(&session_options, "{sv}", "session_handle_token",
                         g_variant_new_string(session_token.c_str()));
    g_variant_builder_add(&session_options, "{sv}", "handle_token",
                         g_variant_new_string(request_token.c_str()));
    
    std::cerr << "[PORTAL] Calling CreateSession with:\n";
    std::cerr << "[PORTAL]   session_handle_token: " << session_token << "\n";
    std::cerr << "[PORTAL]   handle_token: " << request_token << "\n";
    std::cerr << "[PORTAL]   expected request path: " << request_path << "\n";
    
    GVariant *session_result = g_dbus_connection_call_sync(
        connection,
        "org.freedesktop.portal.Desktop",
        "/org/freedesktop/portal/desktop",
        "org.freedesktop.portal.ScreenCast",
        "CreateSession",
        g_variant_new("(a{sv})", &session_options),
        G_VARIANT_TYPE("(o)"),
        G_DBUS_CALL_FLAGS_NONE,
        -1,
        nullptr,
        &error);
    
    if (error) {
        std::cerr << "[PORTAL] CreateSession failed: " << error->message << "\n";
        g_error_free(error);
        g_dbus_connection_signal_unsubscribe(connection, session_signal_id);
        g_object_unref(connection);
        return false;
    }
    
    g_variant_unref(session_result);
    
    // Wait for CreateSession Response
    GMainContext *context = g_main_context_default();
    auto start_time = std::chrono::steady_clock::now();
    while (!session_data.done) {
        g_main_context_iteration(context, FALSE);
        usleep(1000); // 1ms sleep to avoid busy waiting
        
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time).count();
        if (elapsed > 10) {
            std::cerr << "[PORTAL] Timeout waiting for session creation\n";
            g_dbus_connection_signal_unsubscribe(connection, session_signal_id);
            g_object_unref(connection);
            return false;
        }
    }
    
    g_dbus_connection_signal_unsubscribe(connection, session_signal_id);
    
    if (!session_data.success || session_data.session_handle.empty()) {
        std::cerr << "[PORTAL] Failed to create session or get session handle\n";
        g_object_unref(connection);
        return false;
    }
    
    // Use the actual session handle from the portal
    session_handle = session_data.session_handle;
    std::cerr << "[PORTAL] Using session handle: " << session_handle << "\n";
    
    // Step 2: SelectSources - show dialog for user to select screen
    request_token = generate_token();
    request_path = "/org/freedesktop/portal/desktop/request/" + sender_name + "/" + request_token;
    
    GMainLoop *loop_glib = g_main_loop_new(nullptr, FALSE);
    std::atomic<bool> select_done{false};
    
    auto select_response_handler = +[](GDBusConnection *conn, const gchar *sender,
                                       const gchar *object_path, const gchar *interface,
                                       const gchar *signal_name, GVariant *parameters,
                                       gpointer user_data) {
        auto *select_done = static_cast<std::atomic<bool>*>(user_data);
        
        guint32 response;
        GVariant *results;
        g_variant_get(parameters, "(u@a{sv})", &response, &results);
        
        if (response == 0) {
            std::cerr << "[PORTAL] Source selected successfully\n";
        } else {
            std::cerr << "[PORTAL] User cancelled source selection: " << response << "\n";
        }
        
        g_variant_unref(results);
        *select_done = true;
    };
    
    guint select_signal_id = g_dbus_connection_signal_subscribe(
        connection,
        "org.freedesktop.portal.Desktop",
        "org.freedesktop.portal.Request",
        "Response",
        request_path.c_str(),
        nullptr,
        G_DBUS_SIGNAL_FLAGS_NONE,
        select_response_handler,
        &select_done,
        nullptr);
    
    GVariantBuilder source_options;
    g_variant_builder_init(&source_options, G_VARIANT_TYPE_VARDICT);
    g_variant_builder_add(&source_options, "{sv}", "handle_token",
                         g_variant_new_string(request_token.c_str()));
    g_variant_builder_add(&source_options, "{sv}", "types",
                         g_variant_new_uint32(1)); // Monitor=1 only
    g_variant_builder_add(&source_options, "{sv}", "multiple",
                         g_variant_new_boolean(TRUE));
    
    GVariant *select_result = g_dbus_connection_call_sync(
        connection,
        "org.freedesktop.portal.Desktop",
        "/org/freedesktop/portal/desktop",
        "org.freedesktop.portal.ScreenCast",
        "SelectSources",
        g_variant_new("(oa{sv})", session_handle.c_str(), &source_options),
        G_VARIANT_TYPE("(o)"),
        G_DBUS_CALL_FLAGS_NONE,
        -1,
        nullptr,
        &error);
    
    if (error) {
        std::cerr << "[PORTAL] SelectSources failed: " << error->message << "\n";
        g_error_free(error);
        g_dbus_connection_signal_unsubscribe(connection, select_signal_id);
        g_main_loop_unref(loop_glib);
        g_object_unref(connection);
        return false;
    }
    
    g_variant_unref(select_result);
    
    std::cerr << "[PORTAL] Waiting for you to select a screen (60s timeout)...\n";
    
    // Wait for SelectSources response
    start_time = std::chrono::steady_clock::now();
    while (!select_done) {
        g_main_context_iteration(context, FALSE);
        usleep(1000); // 1ms sleep to avoid busy waiting
        
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time).count();
        if (elapsed > 60) {
            std::cerr << "[PORTAL] Timeout waiting for source selection\n";
            g_dbus_connection_signal_unsubscribe(connection, select_signal_id);
            g_main_loop_unref(loop_glib);
            g_object_unref(connection);
            return false;
        }
    }
    
    g_dbus_connection_signal_unsubscribe(connection, select_signal_id);
    
    // Step 3: Start - get PipeWire stream
    request_token = generate_token();
    request_path = "/org/freedesktop/portal/desktop/request/" + sender_name + "/" + request_token;
    
    GVariantBuilder start_options;
    g_variant_builder_init(&start_options, G_VARIANT_TYPE_VARDICT);
    g_variant_builder_add(&start_options, "{sv}", "handle_token",
                         g_variant_new_string(request_token.c_str()));
    
    struct ResponseData {
        PipeWireCapture *cap;
        std::atomic<bool> done{false};
    };
    
    ResponseData response_data;
    response_data.cap = this;
    
    auto response_handler = +[](GDBusConnection *conn, const gchar *sender,
                               const gchar *object_path, const gchar *interface,
                               const gchar *signal_name, GVariant *parameters,
                               gpointer user_data) {
        auto *data = static_cast<ResponseData*>(user_data);
        auto *cap = data->cap;
        
        guint32 response;
        GVariant *results;
        g_variant_get(parameters, "(u@a{sv})", &response, &results);
        
        if (response == 0) {
            // Success - extract PipeWire node ID
            GVariant *streams = g_variant_lookup_value(results, "streams",
                                                       G_VARIANT_TYPE("a(ua{sv})"));
            if (streams) {
                GVariantIter iter;
                g_variant_iter_init(&iter, streams);
                
                guint32 node_id;
                GVariant *stream_properties;
                
                if (g_variant_iter_next(&iter, "(u@a{sv})", &node_id, &stream_properties)) {
                    cap->pipewire_node_id = node_id;
                    std::cerr << "[PORTAL] Got PipeWire node ID: " << node_id << "\n";
                    cap->portal_ready = true;
                    g_variant_unref(stream_properties);
                }
                g_variant_unref(streams);
            }
        } else {
            std::cerr << "[PORTAL] Start failed or cancelled: " << response << "\n";
        }
        
        g_variant_unref(results);
        data->done = true;
    };
    
    guint signal_id = g_dbus_connection_signal_subscribe(
        connection,
        "org.freedesktop.portal.Desktop",
        "org.freedesktop.portal.Request",
        "Response",
        request_path.c_str(),
        nullptr,
        G_DBUS_SIGNAL_FLAGS_NONE,
        response_handler,
        &response_data,
        nullptr);
    
    GVariant *start_result = g_dbus_connection_call_sync(
        connection,
        "org.freedesktop.portal.Desktop",
        "/org/freedesktop/portal/desktop",
        "org.freedesktop.portal.ScreenCast",
        "Start",
        g_variant_new("(osa{sv})", session_handle.c_str(), "", &start_options),
        G_VARIANT_TYPE("(o)"),
        G_DBUS_CALL_FLAGS_NONE,
        -1,
        nullptr,
        &error);
    
    if (error) {
        std::cerr << "[PORTAL] Start failed: " << error->message << "\n";
        g_error_free(error);
        g_dbus_connection_signal_unsubscribe(connection, signal_id);
        g_main_loop_unref(loop_glib);
        g_object_unref(connection);
        return false;
    }
    
    g_variant_unref(start_result);
    
    std::cerr << "[PORTAL] Waiting for stream...\n";
    
    // Wait for Start response
    start_time = std::chrono::steady_clock::now();
    while (!response_data.done) {
        g_main_context_iteration(context, FALSE);
        usleep(1000); // 1ms sleep to avoid busy waiting
        
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time).count();
        if (elapsed > 10) {
            std::cerr << "[PORTAL] Timeout waiting for Start response\n";
            g_dbus_connection_signal_unsubscribe(connection, signal_id);
            g_main_loop_unref(loop_glib);
            g_object_unref(connection);
            return false;
        }
    }
    
    g_dbus_connection_signal_unsubscribe(connection, signal_id);
    g_main_loop_unref(loop_glib);
    
    if (!portal_ready) {
        std::cerr << "[PORTAL] Failed to get PipeWire node\n";
        g_object_unref(connection);
        return false;
    }
    
    portal_connection = connection;
    std::cerr << "[PORTAL] Screen capture ready! (keeping session alive)\n";
    return true;
}

bool PipeWireCapture::init(uint32_t output_index) {
    pw_init(nullptr, nullptr);
    
    std::cerr << "[PW] Initializing capture for output " << output_index << "...\n";
    
    if (!request_screen_cast()) {
        std::cerr << "[PW] Portal request failed for output " << output_index << "\n";
        return false;
    }
    
    if (pipewire_node_id == 0) {
        std::cerr << "[PW] ERROR: No PipeWire node ID received!\n";
        return false;
    }
    
    std::cerr << "[PW] Got node ID: " << pipewire_node_id << "\n";  // ADDED
    
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
    
    uint8_t buffer[8192];
    spa_pod_builder b = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));
    const spa_pod *params[1];

    params[0] = (spa_pod*)spa_pod_builder_add_object(&b,
        SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat,
        SPA_FORMAT_mediaType, SPA_POD_Id(SPA_MEDIA_TYPE_video),
        SPA_FORMAT_mediaSubtype, SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw)
    );
    
    std::cerr << "[PW] Connecting to node " << pipewire_node_id << "...\n";
    
    // CHANGE: Added PW_STREAM_FLAG_DONT_RECONNECT, removed PW_STREAM_FLAG_RT_PROCESS
    int res = pw_stream_connect(stream,
                                PW_DIRECTION_INPUT,
                                pipewire_node_id,
                                static_cast<pw_stream_flags>(
                                    PW_STREAM_FLAG_AUTOCONNECT |
                                    PW_STREAM_FLAG_MAP_BUFFERS |
                                    PW_STREAM_FLAG_DONT_RECONNECT),  // ADDED
                                params, 1);
    
    if (res < 0) {
        std::cerr << "[PW] Failed to connect stream: " << strerror(-res) << "\n";
        pw_thread_loop_unlock(loop);
        return false;
    }
    
    std::cerr << "[PW] Stream connected, starting thread loop...\n";  // ADDED
    
    pw_thread_loop_unlock(loop);
    pw_thread_loop_start(loop);
    
    std::cerr << "[PW] Thread loop started, waiting for stream state...\n";  // ADDED
    
    // CHANGE: Increased timeout to 10 seconds, better state logging
    for (int i = 0; i < 200; i++) {  // Was 100 (5s), now 200 (10s)
        usleep(50000);
        
        pw_thread_loop_lock(loop);
        const char *error = nullptr;
        auto state = pw_stream_get_state(stream, &error);
        if (state == PW_STREAM_STATE_ERROR) {
            std::cerr << "[PW] Stream error state reached: " << (error ? error : "unknown") << "\n";
            pw_thread_loop_unlock(loop);
            return false;
        }

        pw_thread_loop_unlock(loop);
        
        const char *state_names[] = {"error", "unconnected", "connecting", "paused", "streaming"};
        const char *state_name = (state <= PW_STREAM_STATE_STREAMING) ? state_names[state] : "unknown";
        
        if (i == 0 || i % 20 == 0) {
            std::cerr << "[PW] Stream state: " << state_name;
            if (error) std::cerr << " (error: " << error << ")";
            std::cerr << "\n";
        }
        
        if (state == PW_STREAM_STATE_STREAMING) {
            std::cerr << "[PW] *** Stream is STREAMING! ***\n";
            std::cerr << "[PW] Waiting for first frame callback...\n";
            return true;
        }
        
        if (state == PW_STREAM_STATE_ERROR) {
            std::cerr << "[PW] Stream error: " << (error ? error : "unknown") << "\n";
            return false;
        }
    }
    
    pw_thread_loop_lock(loop);
    auto final_state = pw_stream_get_state(stream, nullptr);
    pw_thread_loop_unlock(loop);
    
    std::cerr << "[PW] WARNING: Stream not streaming after 10 seconds (state=" 
              << final_state << ")\n";
    std::cerr << "[PW] Continuing anyway...\n";
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
    
    if (portal_connection) {
        g_object_unref(portal_connection);
        portal_connection = nullptr;
        std::cerr << "[PW] Portal session closed\n";
    }
    
    pw_deinit();
    std::cerr << "[PW] Capture cleaned up\n";
}

bool PipeWireCapture::get_frame(std::vector<uint8_t> &out_frame, uint32_t &out_width,
                                uint32_t &out_height, uint32_t &out_stride) {
    std::lock_guard<std::mutex> lock(mutex);
    
    if (latest_frame.empty() || width == 0 || height == 0) {
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
