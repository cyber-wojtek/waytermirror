#pragma once

#include <linux/input-event-codes.h>
#include <cstdint>

// Alternative virtual input using uinput (works everywhere with proper permissions)
struct UInputDevice {
    int fd = -1;
    bool initialized = false;
    
    bool init_keyboard();
    bool init_mouse();
    void cleanup();
    
    // Keyboard methods
    bool send_key(uint32_t keycode, bool pressed);
    bool send_modifiers(uint32_t mods_depressed, uint32_t mods_locked);
    
    // Mouse methods
    bool send_mouse_move_absolute(int32_t x, int32_t y, uint32_t screen_width, uint32_t screen_height);
    bool send_mouse_button(uint32_t button, bool pressed);
    bool send_mouse_scroll(int32_t direction);
};

struct VirtualInputManager {
    enum Backend {
        WLR_PROTOCOLS,  // wlr-virtual-pointer + zwp-virtual-keyboard
        UINPUT,         // Linux uinput (requires /dev/uinput permissions)
        LIBEI,          // libei (new Wayland standard, future)
        AUTO            // Auto-detect best available
    };
    
    Backend backend = AUTO;
    UInputDevice keyboard_dev;
    UInputDevice mouse_dev;
    
    bool init(Backend preferred = AUTO);
    void cleanup();
    
    // Unified interface
    bool send_key(uint32_t keycode, bool pressed);
    bool send_mouse_move(int32_t x, int32_t y, uint32_t screen_width, uint32_t screen_height);
    bool send_mouse_button(uint32_t button, bool pressed);
    bool send_mouse_scroll(int32_t direction);
    
    const char* backend_name() const;
};

bool uinput_available();
bool wlr_virtual_input_available();
