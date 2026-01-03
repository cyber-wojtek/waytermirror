#pragma once

#include <linux/input-event-codes.h>
#include <cstdint>

// Forward declarations for Wayland types
struct wl_display;
struct wl_seat;
struct zwlr_virtual_pointer_manager_v1;
struct zwp_virtual_keyboard_manager_v1;
struct zwlr_virtual_pointer_v1;
struct zwp_virtual_keyboard_v1;

// uinput backend device
struct UInputDevice {
    int fd = -1;
    bool initialized = false;
    
    bool init_keyboard();
    bool init_mouse();
    void cleanup();
    
    bool send_key(uint32_t keycode, bool pressed);
    bool send_mouse_move_absolute(int32_t x, int32_t y, uint32_t screen_width, uint32_t screen_height);
    bool send_mouse_button(uint32_t button, bool pressed);
    bool send_mouse_scroll(int32_t direction);
};

// WLR protocols backend
struct WLRInputDevice {
    wl_display *display = nullptr;
    wl_seat *seat = nullptr;
    zwlr_virtual_pointer_manager_v1 *pointer_manager = nullptr;
    zwp_virtual_keyboard_manager_v1 *keyboard_manager = nullptr;
    zwlr_virtual_pointer_v1 *virtual_pointer = nullptr;
    zwp_virtual_keyboard_v1 *virtual_keyboard = nullptr;
    bool initialized = false;
    
    // Modifier state
    bool shift_pressed = false;
    bool ctrl_pressed = false;
    bool alt_pressed = false;
    bool super_pressed = false;
    bool altgr_pressed = false;
    bool capslock_on = false;
    bool numlock_on = false;
    
    bool init(wl_display *dpy, wl_seat *st,
              zwlr_virtual_pointer_manager_v1 *ptr_mgr,
              zwp_virtual_keyboard_manager_v1 *kbd_mgr);
    void cleanup();
    void setup_keymap();
    
    bool send_key(uint32_t keycode, bool pressed);
    bool send_mouse_move(int32_t x, int32_t y, uint32_t screen_width, uint32_t screen_height);
    bool send_mouse_button(uint32_t button, bool pressed);
    bool send_mouse_scroll(int32_t direction);
};

struct VirtualInputManager {
    enum Backend {
        NONE = 0,
        WLR_PROTOCOLS,
        UINPUT,
        LIBEI,
        AUTO
    };
    
    Backend backend = NONE;
    UInputDevice uinput_kbd;
    UInputDevice uinput_mouse;
    WLRInputDevice wlr;
    
    // Initialize WLR backend
    bool init_wlr(wl_display *dpy, wl_seat *seat,
                  zwlr_virtual_pointer_manager_v1 *ptr_mgr,
                  zwp_virtual_keyboard_manager_v1 *kbd_mgr);
    
    // Initialize uinput backend
    bool init_uinput();
    
    void cleanup();
    
    // Unified interface
    bool send_key(uint32_t keycode, bool pressed);
    bool send_mouse_move(int32_t x, int32_t y, uint32_t screen_width, uint32_t screen_height);
    bool send_mouse_button(uint32_t button, bool pressed);
    bool send_mouse_scroll(int32_t direction);
    
    const char* backend_name() const;
    bool is_initialized() const { return backend != NONE; }
};

bool uinput_available();
