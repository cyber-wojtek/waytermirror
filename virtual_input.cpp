#include "virtual_input.h"
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <chrono>
#include <linux/uinput.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <wayland-client.h>
#include "wlr-virtual-pointer-unstable-v1-client-protocol.h"
#include "virtual-keyboard-unstable-v1-client-protocol.h"

// Helper for shared memory
static int create_shm_file(size_t size) {
    char name[] = "/wl_shm-XXXXXX";
    int fd = memfd_create(name, MFD_CLOEXEC);
    if (fd < 0) return -1;
    if (ftruncate(fd, size) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

bool UInputDevice::init_keyboard() {
    fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK);
    if (fd < 0) {
        std::cerr << "[UINPUT] Failed to open /dev/uinput (may need permissions)\n";
        std::cerr << "[UINPUT] Try: sudo chmod 666 /dev/uinput\n";
        return false;
    }
    
    // Enable key events
    ioctl(fd, UI_SET_EVBIT, EV_KEY);
    ioctl(fd, UI_SET_EVBIT, EV_SYN);
    
    // Enable all keyboard keys (0-255 covers most keys)
    for (int i = 0; i < 256; i++) {
        ioctl(fd, UI_SET_KEYBIT, i);
    }
    
    // Setup device
    struct uinput_setup usetup = {};
    strncpy(usetup.name, "Waytermirror Virtual Keyboard", UINPUT_MAX_NAME_SIZE - 1);
    usetup.id.bustype = BUS_VIRTUAL;
    usetup.id.vendor = 0x1234;
    usetup.id.product = 0x5678;
    usetup.id.version = 1;
    
    if (ioctl(fd, UI_DEV_SETUP, &usetup) < 0) {
        std::cerr << "[UINPUT] Failed to setup keyboard device\n";
        close(fd);
        fd = -1;
        return false;
    }
    
    if (ioctl(fd, UI_DEV_CREATE) < 0) {
        std::cerr << "[UINPUT] Failed to create keyboard device\n";
        close(fd);
        fd = -1;
        return false;
    }
    
    initialized = true;
    std::cerr << "[UINPUT] Virtual keyboard created\n";
    return true;
}

bool UInputDevice::init_mouse() {
    fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK);
    if (fd < 0) {
        std::cerr << "[UINPUT] Failed to open /dev/uinput (may need permissions)\n";
        return false;
    }
    
    // Enable events
    ioctl(fd, UI_SET_EVBIT, EV_KEY);
    ioctl(fd, UI_SET_EVBIT, EV_ABS);
    ioctl(fd, UI_SET_EVBIT, EV_REL);
    ioctl(fd, UI_SET_EVBIT, EV_SYN);
    
    // Mouse buttons
    ioctl(fd, UI_SET_KEYBIT, BTN_LEFT);
    ioctl(fd, UI_SET_KEYBIT, BTN_RIGHT);
    ioctl(fd, UI_SET_KEYBIT, BTN_MIDDLE);
    
    // Absolute positioning
    struct uinput_abs_setup abs_setup_x = {};
    abs_setup_x.code = ABS_X;
    abs_setup_x.absinfo.minimum = 0;
    abs_setup_x.absinfo.maximum = 65535;
    ioctl(fd, UI_ABS_SETUP, &abs_setup_x);
    
    struct uinput_abs_setup abs_setup_y = {};
    abs_setup_y.code = ABS_Y;
    abs_setup_y.absinfo.minimum = 0;
    abs_setup_y.absinfo.maximum = 65535;
    ioctl(fd, UI_ABS_SETUP, &abs_setup_y);
    
    // Scroll wheel
    ioctl(fd, UI_SET_RELBIT, REL_WHEEL);
    
    // Setup device
    struct uinput_setup usetup = {};
    strncpy(usetup.name, "Waytermirror Virtual Mouse", UINPUT_MAX_NAME_SIZE - 1);
    usetup.id.bustype = BUS_VIRTUAL;
    usetup.id.vendor = 0x1234;
    usetup.id.product = 0x5679;
    usetup.id.version = 1;
    
    if (ioctl(fd, UI_DEV_SETUP, &usetup) < 0) {
        std::cerr << "[UINPUT] Failed to setup mouse device\n";
        close(fd);
        fd = -1;
        return false;
    }
    
    if (ioctl(fd, UI_DEV_CREATE) < 0) {
        std::cerr << "[UINPUT] Failed to create mouse device\n";
        close(fd);
        fd = -1;
        return false;
    }
    
    initialized = true;
    std::cerr << "[UINPUT] Virtual mouse created\n";
    return true;
}

void UInputDevice::cleanup() {
    if (fd >= 0) {
        if (initialized) {
            ioctl(fd, UI_DEV_DESTROY);
        }
        close(fd);
        fd = -1;
    }
    initialized = false;
}

bool UInputDevice::send_key(uint32_t keycode, bool pressed) {
    if (fd < 0) return false;
    
    struct input_event ev = {};
    ev.type = EV_KEY;
    ev.code = keycode;
    ev.value = pressed ? 1 : 0;
    
    if (write(fd, &ev, sizeof(ev)) < 0) {
        return false;
    }
    
    // Sync event
    ev.type = EV_SYN;
    ev.code = SYN_REPORT;
    ev.value = 0;
    write(fd, &ev, sizeof(ev));
    
    return true;
}

bool UInputDevice::send_mouse_move_absolute(int32_t x, int32_t y, 
                                            uint32_t screen_width, uint32_t screen_height) {
    if (fd < 0) return false;
    
    // Map screen coordinates to 0-65535 range
    int32_t abs_x = (x * 65535) / screen_width;
    int32_t abs_y = (y * 65535) / screen_height;
    
    struct input_event ev = {};
    
    // X position
    ev.type = EV_ABS;
    ev.code = ABS_X;
    ev.value = abs_x;
    write(fd, &ev, sizeof(ev));
    
    // Y position
    ev.code = ABS_Y;
    ev.value = abs_y;
    write(fd, &ev, sizeof(ev));
    
    // Sync
    ev.type = EV_SYN;
    ev.code = SYN_REPORT;
    ev.value = 0;
    write(fd, &ev, sizeof(ev));
    
    return true;
}

bool UInputDevice::send_mouse_button(uint32_t button, bool pressed) {
    if (fd < 0) return false;
    
    struct input_event ev = {};
    ev.type = EV_KEY;
    ev.code = button; // BTN_LEFT, BTN_RIGHT, BTN_MIDDLE
    ev.value = pressed ? 1 : 0;
    
    write(fd, &ev, sizeof(ev));
    
    // Sync
    ev.type = EV_SYN;
    ev.code = SYN_REPORT;
    ev.value = 0;
    write(fd, &ev, sizeof(ev));
    
    return true;
}

bool UInputDevice::send_mouse_scroll(int32_t direction) {
    if (fd < 0) return false;
    
    struct input_event ev = {};
    ev.type = EV_REL;
    ev.code = REL_WHEEL;
    ev.value = direction;
    
    write(fd, &ev, sizeof(ev));
    
    ev.type = EV_SYN;
    ev.code = SYN_REPORT;
    ev.value = 0;
    write(fd, &ev, sizeof(ev));
    
    return true;
}

// WLRInputDevice implementation

bool WLRInputDevice::init(wl_display *dpy, wl_seat *st,
                          zwlr_virtual_pointer_manager_v1 *ptr_mgr,
                          zwp_virtual_keyboard_manager_v1 *kbd_mgr) {
    if (!dpy || !st || !ptr_mgr || !kbd_mgr) {
        std::cerr << "[WLR] Missing required Wayland objects\n";
        return false;
    }
    
    display = dpy;
    seat = st;
    pointer_manager = ptr_mgr;
    keyboard_manager = kbd_mgr;
    
    virtual_pointer = zwlr_virtual_pointer_manager_v1_create_virtual_pointer(pointer_manager, seat);
    if (!virtual_pointer) {
        std::cerr << "[WLR] Failed to create virtual pointer\n";
        return false;
    }
    
    virtual_keyboard = zwp_virtual_keyboard_manager_v1_create_virtual_keyboard(keyboard_manager, seat);
    if (!virtual_keyboard) {
        std::cerr << "[WLR] Failed to create virtual keyboard\n";
        zwlr_virtual_pointer_v1_destroy(virtual_pointer);
        virtual_pointer = nullptr;
        return false;
    }
    
    setup_keymap();
    initialized = true;
    std::cerr << "[WLR] Virtual input devices created\n";
    return true;
}

void WLRInputDevice::cleanup() {
    if (virtual_keyboard) {
        zwp_virtual_keyboard_v1_destroy(virtual_keyboard);
        virtual_keyboard = nullptr;
    }
    if (virtual_pointer) {
        zwlr_virtual_pointer_v1_destroy(virtual_pointer);
        virtual_pointer = nullptr;
    }
    initialized = false;
}

void WLRInputDevice::setup_keymap() {
    if (!virtual_keyboard) return;
    
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
    if (fd < 0) {
        std::cerr << "[WLR] Failed to create keymap shm\n";
        return;
    }
    
    void *data = mmap(nullptr, keymap_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
        close(fd);
        return;
    }
    
    memcpy(data, keymap_str.c_str(), keymap_size);
    munmap(data, keymap_size);
    
    zwp_virtual_keyboard_v1_keymap(virtual_keyboard, WL_KEYBOARD_KEYMAP_FORMAT_XKB_V1, fd, keymap_size);
    close(fd);
    wl_display_flush(display);
    
    std::cerr << "[WLR] Keymap configured\n";
}

bool WLRInputDevice::send_key(uint32_t keycode, bool pressed) {
    if (!virtual_keyboard) return false;
    
    uint32_t time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Track modifier state
    bool is_shift = (keycode == KEY_LEFTSHIFT || keycode == KEY_RIGHTSHIFT);
    bool is_ctrl = (keycode == KEY_LEFTCTRL || keycode == KEY_RIGHTCTRL);
    bool is_alt = (keycode == KEY_LEFTALT || keycode == KEY_RIGHTALT);
    bool is_super = (keycode == KEY_LEFTMETA || keycode == KEY_RIGHTMETA);
    bool is_altgr = (keycode == KEY_RIGHTALT);
    bool is_capslock = (keycode == KEY_CAPSLOCK);
    bool is_numlock = (keycode == KEY_NUMLOCK);
    
    if (is_shift) shift_pressed = pressed;
    if (is_ctrl) ctrl_pressed = pressed;
    if (is_alt) alt_pressed = pressed;
    if (is_super) super_pressed = pressed;
    if (is_altgr) altgr_pressed = pressed;
    if (pressed && is_capslock) capslock_on = !capslock_on;
    if (pressed && is_numlock) numlock_on = !numlock_on;
    
    uint32_t state = pressed ? WL_KEYBOARD_KEY_STATE_PRESSED : WL_KEYBOARD_KEY_STATE_RELEASED;
    zwp_virtual_keyboard_v1_key(virtual_keyboard, time_ms, keycode, state);
    
    uint32_t mods_depressed = 0;
    uint32_t mods_locked = 0;
    
    if (shift_pressed) mods_depressed |= (1 << 0);
    if (ctrl_pressed) mods_depressed |= (1 << 2);
    if (alt_pressed) mods_depressed |= (1 << 3);
    if (super_pressed) mods_depressed |= (1 << 6);
    if (altgr_pressed) mods_depressed |= (1 << 7);
    if (capslock_on) mods_locked |= (1 << 1);
    if (numlock_on) mods_locked |= (1 << 4);
    
    zwp_virtual_keyboard_v1_modifiers(virtual_keyboard, mods_depressed, 0, mods_locked, 0);
    wl_display_flush(display);
    
    return true;
}

bool WLRInputDevice::send_mouse_move(int32_t x, int32_t y, uint32_t screen_width, uint32_t screen_height) {
    if (!virtual_pointer) return false;
    
    uint32_t time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    zwlr_virtual_pointer_v1_motion_absolute(virtual_pointer, time_ms,
                                            (uint32_t)x, (uint32_t)y,
                                            screen_width, screen_height);
    zwlr_virtual_pointer_v1_frame(virtual_pointer);
    wl_display_flush(display);
    
    return true;
}

bool WLRInputDevice::send_mouse_button(uint32_t button, bool pressed) {
    if (!virtual_pointer) return false;
    
    uint32_t time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    zwlr_virtual_pointer_v1_button(virtual_pointer, time_ms, button,
                                   pressed ? WL_POINTER_BUTTON_STATE_PRESSED 
                                           : WL_POINTER_BUTTON_STATE_RELEASED);
    zwlr_virtual_pointer_v1_frame(virtual_pointer);
    wl_display_flush(display);
    
    return true;
}

bool WLRInputDevice::send_mouse_scroll(int32_t direction) {
    if (!virtual_pointer) return false;
    
    uint32_t time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    wl_fixed_t value = wl_fixed_from_int(direction * 15);
    zwlr_virtual_pointer_v1_axis(virtual_pointer, time_ms, WL_POINTER_AXIS_VERTICAL_SCROLL, value);
    zwlr_virtual_pointer_v1_frame(virtual_pointer);
    wl_display_flush(display);
    
    return true;
}

// VirtualInputManager implementation

bool VirtualInputManager::init_wlr(wl_display *dpy, wl_seat *seat,
                                   zwlr_virtual_pointer_manager_v1 *ptr_mgr,
                                   zwp_virtual_keyboard_manager_v1 *kbd_mgr) {
    if (backend != NONE) {
        std::cerr << "[INPUT] Already initialized with " << backend_name() << "\n";
        return false;
    }
    
    if (wlr.init(dpy, seat, ptr_mgr, kbd_mgr)) {
        backend = WLR_PROTOCOLS;
        std::cerr << "[INPUT] Initialized WLR protocols backend\n";
        return true;
    }
    return false;
}

bool VirtualInputManager::init_uinput() {
    if (backend != NONE) {
        std::cerr << "[INPUT] Already initialized with " << backend_name() << "\n";
        return false;
    }
    
    if (!uinput_kbd.init_keyboard()) {
        std::cerr << "[INPUT] Failed to init uinput keyboard\n";
        return false;
    }
    
    if (!uinput_mouse.init_mouse()) {
        std::cerr << "[INPUT] Failed to init uinput mouse\n";
        uinput_kbd.cleanup();
        return false;
    }
    
    backend = UINPUT;
    std::cerr << "[INPUT] Initialized uinput backend\n";
    return true;
}

void VirtualInputManager::cleanup() {
    if (backend == UINPUT) {
        uinput_kbd.cleanup();
        uinput_mouse.cleanup();
    } else if (backend == WLR_PROTOCOLS) {
        wlr.cleanup();
    }
    backend = NONE;
}

bool VirtualInputManager::send_key(uint32_t keycode, bool pressed) {
    switch (backend) {
        case UINPUT: return uinput_kbd.send_key(keycode, pressed);
        case WLR_PROTOCOLS: return wlr.send_key(keycode, pressed);
        default: return false;
    }
}

bool VirtualInputManager::send_mouse_move(int32_t x, int32_t y, 
                                          uint32_t screen_width, uint32_t screen_height) {
    switch (backend) {
        case UINPUT: return uinput_mouse.send_mouse_move_absolute(x, y, screen_width, screen_height);
        case WLR_PROTOCOLS: return wlr.send_mouse_move(x, y, screen_width, screen_height);
        default: return false;
    }
}

bool VirtualInputManager::send_mouse_button(uint32_t button, bool pressed) {
    switch (backend) {
        case UINPUT: return uinput_mouse.send_mouse_button(button, pressed);
        case WLR_PROTOCOLS: return wlr.send_mouse_button(button, pressed);
        default: return false;
    }
}

bool VirtualInputManager::send_mouse_scroll(int32_t direction) {
    switch (backend) {
        case UINPUT: return uinput_mouse.send_mouse_scroll(direction);
        case WLR_PROTOCOLS: return wlr.send_mouse_scroll(direction);
        default: return false;
    }
}

const char* VirtualInputManager::backend_name() const {
    switch (backend) {
        case WLR_PROTOCOLS: return "WLR Protocols";
        case UINPUT: return "uinput";
        case LIBEI: return "libei";
        case NONE: return "none";
        default: return "unknown";
    }
}

bool uinput_available() {
    int fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK);
    if (fd < 0) return false;
    close(fd);
    return true;
}