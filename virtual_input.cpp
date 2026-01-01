#include "virtual_input.h"
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <linux/uinput.h>
#include <sys/ioctl.h>

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

bool UInputDevice::send_modifiers(uint32_t mods_depressed, uint32_t mods_locked) {
    // uinput handles modifiers through key events, so this is handled
    // by send_key() calls for modifier keys
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
    
    // Sync
    ev.type = EV_SYN;
    ev.code = SYN_REPORT;
    ev.value = 0;
    write(fd, &ev, sizeof(ev));
    
    return true;
}

// VirtualInputManager implementation

bool VirtualInputManager::init(Backend preferred) {
    backend = preferred;
    
    if (backend == AUTO) {
        // Auto-detect best available backend
        if (wlr_virtual_input_available()) {
            backend = WLR_PROTOCOLS;
            std::cerr << "[INPUT] Using WLR virtual input protocols\n";
            return true; // Will use existing global virtual_keyboard/virtual_pointer
        } else if (uinput_available()) {
            backend = UINPUT;
        } else {
            std::cerr << "[INPUT] No virtual input backend available!\n";
            return false;
        }
    }
    
    if (backend == UINPUT) {
        std::cerr << "[INPUT] Using uinput backend\n";
        
        if (!keyboard_dev.init_keyboard()) {
            std::cerr << "[INPUT] Failed to initialize uinput keyboard\n";
            return false;
        }
        
        if (!mouse_dev.init_mouse()) {
            std::cerr << "[INPUT] Failed to initialize uinput mouse\n";
            keyboard_dev.cleanup();
            return false;
        }
        
        return true;
    }
    
    return false;
}

void VirtualInputManager::cleanup() {
    if (backend == UINPUT) {
        keyboard_dev.cleanup();
        mouse_dev.cleanup();
    }
}

bool VirtualInputManager::send_key(uint32_t keycode, bool pressed) {
    if (backend == UINPUT) {
        return keyboard_dev.send_key(keycode, pressed);
    }
    return false;
}

bool VirtualInputManager::send_mouse_move(int32_t x, int32_t y, 
                                         uint32_t screen_width, uint32_t screen_height) {
    if (backend == UINPUT) {
        return mouse_dev.send_mouse_move_absolute(x, y, screen_width, screen_height);
    }
    return false;
}

bool VirtualInputManager::send_mouse_button(uint32_t button, bool pressed) {
    if (backend == UINPUT) {
        return mouse_dev.send_mouse_button(button, pressed);
    }
    return false;
}

bool VirtualInputManager::send_mouse_scroll(int32_t direction) {
    if (backend == UINPUT) {
        return mouse_dev.send_mouse_scroll(direction);
    }
    return false;
}

const char* VirtualInputManager::backend_name() const {
    switch (backend) {
        case WLR_PROTOCOLS: return "WLR Protocols";
        case UINPUT: return "uinput";
        case LIBEI: return "libei";
        case AUTO: return "auto";
        default: return "unknown";
    }
}

bool uinput_available() {
    int fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK);
    if (fd < 0) {
        return false;
    }
    close(fd);
    return true;
}

bool wlr_virtual_input_available() {
    // This will be checked in main code based on Wayland protocol availability
    // For now, just return false to force uinput detection
    return false;
}
